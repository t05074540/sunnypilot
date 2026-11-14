"""Altis longitudinal tune.

A port of the driving-behaviour half of henryopen/openpilot (Custin openpilot, branch c4-only,
2026-11-06) onto sunnypilot's planner, for a 12th generation Toyota Corolla Altis (TSS 2.0).

What is here, all of it under the AltisLongTune param and only on a Toyota with openpilot
longitudinal control. Each group has its own switch (AltisSettings) so they can be tried one at a time:

  cruise candidate (cruise_accel)
    - an acceleration ceiling that depends on speed and on whether the road ahead is clear
    - a higher ceiling behind a lead that is plainly pulling away ("chase")
    - a lateral/longitudinal budget, so it does not accelerate hard while cornering
    - a sqrt-shaped approach to the set speed with a small deadzone, instead of a proportional law
    - curve_speed: ease off before a corner the model can see
    - no speeding up towards a lead the model is unsure about ("weak lead")

  planner output (shape_output)
    - no throttle towards a lead already being closed on ("coast")
    - no closing on a new lead before its speed is known ("cut")
    - braking for cruise or a lead builds up rather than arriving at once ("brake onset")

What is deliberately NOT here, and why:

  - junction handoff, stop_for_lights, light_ahead, the keypad's GO: sunnypilot's planner always
    lets the driving model compete (Autospeed), so the model already stops for lights itself.
  - lead_stop, long_deadzone, dash_to_true and most of standstill_hold: fitted to the Custin's own
    actuator, stop request and speedometer. They describe that car, not this one. Only the latch of
    standstill_hold is here (standstill_hold.py), off by default.
  - lead_view: it rewrites the lead the MPC sees.
  - turn_slow, road_memory: they need the fork's own map daemon.
  - everything lateral and everything in opendbc/panda: Toyota's stock limits are left alone.

Every number below was measured on a Hyundai Custin by its one driver. They are a starting point
for an Altis, not a calibration. Nothing in this file can ask for more deceleration than the
planner already wanted, with two exceptions: A_CRUISE_MIN lets the cruise candidate slow to a
lowered set speed at up to 1.2 m/s^2 where stock stops at 0.5, and the lane-change gap (off by
default) follows closer than the personality's gap while a lane change is under way.
"""
from dataclasses import dataclass
import math

import numpy as np

from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.altis.curve_speed import CurveSpeedControl
from openpilot.sunnypilot.selfdrive.controls.lib.altis.standstill_hold import StandstillHold

# ---- settings ------------------------------------------------------------------------------------
ACCEL_PROFILE_SCALE = (0.8, 1.0, 1.25)      # gentle, standard, brisk: on every acceleration ceiling below
CURVE_COMFORT_SCALE = (0.85, 1.0, 1.15)     # slower, standard, faster: on curve_speed's lateral comfort curve


@dataclass
class AltisSettings:
  accel_curve: bool = True       # the ceiling and the tapered approach to the set speed
  accel_profile: int = 1         # index into ACCEL_PROFILE_SCALE
  curve_speed: bool = True
  curve_comfort: int = 1         # index into CURVE_COMFORT_SCALE
  lead_shaping: bool = True      # weak lead, coast, cut
  brake_onset: bool = True
  lane_change_gap: bool = False
  standstill_hold: bool = False


class Limit:
  """What last held the acceleration back. Mirrors custom.LongitudinalPlanSP.Altis.Limit."""
  none = 0
  free = 1         # the clear-road ceiling
  lead = 2         # the ceiling with a lead in reach
  chase = 3        # the raised ceiling behind a leaving lead
  lateral = 4      # the cornering budget
  curve = 5        # curve_speed
  weakLead = 6
  coast = 7
  cut = 8
  brakeOnset = 9
  hold = 10


# the same, by name, for writing the message
LIMIT_NAMES = ("none", "free", "lead", "chase", "lateral", "curve", "weakLead", "coast", "cut", "brakeOnset", "hold")

# ---- cruise ceiling ------------------------------------------------------------------------------
#                     0    10km/h  18    36    54    72    90   144
A_CRUISE_MAX_BP =   [0.,   2.8,   5.,   10.,  15.,  20.,  25., 40.]
A_CRUISE_MAX_VALS = [0.72, 0.68,  0.78, 0.48, 0.40, 0.38, 0.34, 0.26]       # a lead within reach
A_CRUISE_MAX_VALS_FREE = [0.72, 0.80, 0.95, 0.80, 0.75, 0.75, 0.64, 0.50]   # the road ahead clear
FREE_LEAD_MARGIN = 1.2       # of the MPC's own target gap: beyond it the road counts as clear

CHASE_DV_PLAN = 0.5          # m/s the lead is faster than us by
CHASE_DV = 1.5               # m/s, the radar's own test (is_chase)
CHASE_A = 0.2                # m/s^2
CHASE_BP = [0., 2.5, 5., 10., 15., 20., 25., 40.]
CHASE_VALS = [1.2, 1.2, 1.14, 0.85, 0.80, 0.70, 0.64, 0.50]

J_CRUISE_BP = [0., 10.0, 25., 40.]
J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MIN = -1.2
J_CRUISE_COMFORT = 0.16      # m/s^3, the taper into the set speed
V_CRUISE_DEADZONE = 0.25     # m/s

_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

# ---- a lead the model is unsure about ------------------------------------------------------------
RADAR_TO_CAMERA = 1.52
WEAK_LEAD_MIN_PROB, WEAK_LEAD_MAX_PROB = 0.2, 0.5
WEAK_LEAD_MIN_FRAMES = 10
WEAK_LEAD_MIN_DIST, WEAK_LEAD_MAX_DIST = 45., 100.
WEAK_LEAD_MIN_SPEED = 5.

# ---- planner output ------------------------------------------------------------------------------
COAST_TTC = 15.0             # s, 0 = off
COAST_CLOSING = 0.3          # m/s

CUT_JUMP_M = 6.0             # m: the lead's range growing by more than this in a frame = a new car
CUT_S = 5.0                  # s
CUT_A = 0.2                  # m/s^2
CUT_AWAY = 1.0               # m/s the range opens by on average since the jump: the new car is leaving
CUT_AWAY_MIN_S = 0.5         # s of range before that is believed
CUT_MIN_V = 10 * CV.KPH_TO_MS
CUT_LOST_S = 2.0             # s with no lead at all: the hold is over

BRAKE_ONSET_J = 1.5          # m/s^3, 0 = off
BRAKE_ONSET_TTC = 3.0        # s
BRAKE_ONSET_GAP = 0.7        # of the follow distance
BRAKE_ONSET_MIN_V = 10 * CV.KPH_TO_MS

# ---- a shorter follow gap while a lane change is under way ---------------------------------------
# Moving across, the MPC meets the next lane's car inside its follow gap and brakes to open it. So
# while laneChangeStarting/Finishing the gap comes down to LC_T_FOLLOW at LC_RAMP_IN /s and goes back
# at LC_RAMP_OUT /s; at once if the lead brakes LC_ABORT_A or harder, the blind spot on that side is
# taken, a pedal is pressed, or under LC_MIN_V.
LC_T_FOLLOW = 1.0            # s
LC_RAMP_IN = 0.6             # s of t_follow per s
LC_RAMP_OUT = 0.1            # s of t_follow per s
LC_ABORT_A = -0.8            # m/s^2, the lead's aLeadK
LC_MIN_V = 30 * CV.KPH_TO_MS
LC_STATES = (log.LaneChangeState.laneChangeStarting, log.LaneChangeState.laneChangeFinishing)


def is_chase(lead) -> bool:
  """A lead that is leaving. Radar only: vision's speed is too noisy to say so."""
  if not lead.present or not lead.radar:
    return False
  v_ego = float(lead.vLead) - float(lead.vRel)
  return float(lead.vLead) > v_ego + CHASE_DV and float(lead.aLeadK) > CHASE_A


class AltisLongTune:
  def __init__(self, CP, comfort_brake: float, stop_distance: float, dt: float = DT_MDL):
    self.dt = dt
    self.comfort_brake = comfort_brake
    self.stop_distance = stop_distance
    self.supported = bool(CP.openpilotLongitudinalControl) and CP.brand == "toyota"
    self.enabled = False       # the AltisLongTune param, set by the planner
    self.settings = AltisSettings()

    self.curve_speed = CurveSpeedControl()
    self.weak_lead_frames = 0
    self.follow_distance = 0.
    self.a_cruise_max = 0.
    self.curve_limited = False
    self.weak_lead = False
    self.limit = Limit.none
    self.t_follow = 0.
    self.lc_t_follow = None    # the follow gap the lane-change ramp is at, None when not in use
    self.hold = StandstillHold(dt)
    self.hold_active = False

    self.cut_frames = 0        # frames left of the hold after the lead became a different car
    self.cut_d0 = 0.
    self.cut_since = 0
    self.cut_prev_d = -1.
    self.cut_lost = 0          # frames in a row with no lead at all

  @property
  def active(self) -> bool:
    return self.enabled and self.supported

  # the MPC's own distances, in its own terms
  def _safe_obstacle_distance(self, v_ego: float, t_follow: float) -> float:
    return v_ego ** 2 / (2 * self.comfort_brake) + t_follow * v_ego + self.stop_distance

  def _stopped_equivalence(self, v_lead: float) -> float:
    return v_lead ** 2 / (2 * self.comfort_brake)

  def _lead_is_far(self, lead, v_ego: float, t_follow: float) -> bool:
    """Is the nearest lead far enough that cruise, not the MPC, is what limits acceleration?"""
    if not lead.present:
      return True
    target_gap = self._safe_obstacle_distance(v_ego, t_follow) - self._stopped_equivalence(min(float(lead.vLead), v_ego))
    return float(lead.dRel) > FREE_LEAD_MARGIN * target_gap

  def lane_change_t_follow(self, sm, base: float, v_ego: float, reset_state: bool) -> float:
    """The MPC's follow gap: the personality's, or less while a lane change is under way - see LC_*."""
    if not self.settings.lane_change_gap:
      self.lc_t_follow = None
      return base
    meta, cs, lead = sm['modelV2'].meta, sm['carState'], sm['radarState'].leadOne
    direction = meta.laneChangeDirection
    cur = base if self.lc_t_follow is None else min(self.lc_t_follow, base)
    if (reset_state or cs.gasPressed or cs.brakePressed or v_ego < LC_MIN_V
        or (direction == log.LaneChangeDirection.left and cs.leftBlindspot)
        or (direction == log.LaneChangeDirection.right and cs.rightBlindspot)
        or (lead.present and float(lead.aLeadK) <= LC_ABORT_A)):
      cur = base
    elif meta.laneChangeState in LC_STATES:
      cur = max(min(LC_T_FOLLOW, base), cur - LC_RAMP_IN * self.dt)
    else:
      cur = min(base, cur + LC_RAMP_OUT * self.dt)
    self.lc_t_follow = None if cur >= base else cur
    return cur

  def cruise_accel(self, sm, v_cruise: float, v_ego: float, a_cruise_prev: float, t_follow: float,
                   long_active: bool, a_cruise_stock: float) -> float:
    """The cruise candidate. a_cruise_stock is the planner's own, used when the accel curve is switched off."""
    cs = sm['carState']
    lead = sm['radarState'].leadOne
    st = self.settings
    self.t_follow = t_follow
    self.limit = Limit.none
    scale = ACCEL_PROFILE_SCALE[int(np.clip(st.accel_profile, 0, len(ACCEL_PROFILE_SCALE) - 1))]

    self.follow_distance = (self._safe_obstacle_distance(v_ego, t_follow)
                            - self._stopped_equivalence(float(lead.vLead))) if lead.present else 0.

    lead_free = self._lead_is_far(lead, v_ego, t_follow)
    max_accel = scale * float(np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS_FREE if lead_free else A_CRUISE_MAX_VALS))
    ceiling = Limit.free if lead_free else Limit.lead
    if (lead.present and float(lead.vLead) > v_ego + CHASE_DV_PLAN
        and float(lead.dRel) >= self.follow_distance and is_chase(lead)):
      chase_accel = scale * float(np.interp(v_ego, CHASE_BP, CHASE_VALS))
      if chase_accel > max_accel:
        max_accel, ceiling = chase_accel, Limit.chase

    # not hard out of a corner: what is left of the budget once the sideways part is taken
    a_total_max = float(np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V))
    a_y = v_ego ** 2 * float(sm['controlsState'].curvature)
    a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
    if a_x_allowed < max_accel - 1e-4:
      max_accel, ceiling = a_x_allowed, Limit.lateral
    self.a_cruise_max = max_accel

    # taper into the set speed rather than holding full accel or decel to the last few km/h,
    # and ignore the last fraction of a km/h so holding a speed is not a constant correction
    speed_error = v_cruise - v_ego
    if abs(speed_error) <= V_CRUISE_DEADZONE:
      speed_error = 0.0
    else:
      speed_error -= math.copysign(V_CRUISE_DEADZONE, speed_error)
    comfort_accel = min(abs(speed_error), math.sqrt(2. * J_CRUISE_COMFORT * abs(speed_error)))
    target = float(np.clip(math.copysign(comfort_accel, speed_error), A_CRUISE_MIN, max_accel))
    j_cruise = float(np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS))
    a_cruise = float(np.clip(target, a_cruise_prev - j_cruise * self.dt, a_cruise_prev + j_cruise * self.dt))
    if not st.accel_curve:
      a_cruise, self.a_cruise_max = a_cruise_stock, 0.
    elif a_cruise >= max_accel - 1e-4 and speed_error > 0.:
      self.limit = ceiling     # sitting on the ceiling: it, not the set speed, is what holds the car back

    # ease off before a corner the model can see
    self.curve_speed.v2.lat_scale = CURVE_COMFORT_SCALE[int(np.clip(st.curve_comfort, 0, len(CURVE_COMFORT_SCALE) - 1))]
    self.curve_speed.update(sm, long_active and st.curve_speed, cs.gasPressed, v_ego, cs.aEgo)
    self.curve_limited = st.curve_speed and self.curve_speed.is_active and self.curve_speed.a_target < a_cruise
    if self.curve_limited:
      a_cruise = min(a_cruise, float(self.curve_speed.a_target))
      self.limit = Limit.curve

    # a lead the model is unsure about, too far out to hold as a lead: do not speed up at it.
    # The onset has to last, the release does not.
    weak_now = False
    if not lead.present and v_ego > WEAK_LEAD_MIN_SPEED:
      leads = sm['modelV2'].leadsV3
      if len(leads):
        lead_x = leads[0].x[0] - RADAR_TO_CAMERA
        weak_now = (WEAK_LEAD_MIN_PROB <= leads[0].prob < WEAK_LEAD_MAX_PROB
                    and WEAK_LEAD_MIN_DIST <= lead_x < WEAK_LEAD_MAX_DIST)
    self.weak_lead_frames = self.weak_lead_frames + 1 if weak_now else 0
    self.weak_lead = st.lead_shaping and self.weak_lead_frames >= WEAK_LEAD_MIN_FRAMES
    if self.weak_lead and a_cruise > 0.:
      a_cruise = 0.
      self.limit = Limit.weakLead

    return a_cruise

  def shape_output(self, sm, output_a_target: float, a_prev: float, v_ego: float, reset_state: bool,
                   model_led: bool, should_stop: bool, fcw: bool, cruise_led: bool = True) -> float:
    """After the candidates are resolved. model_led / cruise_led: whose plan won the min().

    The hold, coast and cut only ever take acceleration away. The brake onset lets braking arrive later,
    so it stands down for anything urgent and for anything the model or a stop asked for."""
    st = self.settings
    lead = sm['radarState'].leadOne
    present = bool(lead.present)
    d_now = float(lead.dRel) if present else -1.
    if not cruise_led:
      self.limit = Limit.none   # what cruise_accel found only matters while cruise is the one in charge

    # stopped behind a car that has not gone: stay stopped until it has moved on its own
    if st.standstill_hold:
      self.hold_active = self.hold.update(not reset_state, v_ego, lead, sm['carState'].gasPressed)
    else:
      self.hold.reset()
      self.hold_active = False
    if self.hold_active:
      output_a_target = min(output_a_target, 0.)
      self.limit = Limit.hold

    # no throttle towards a lead we are already closing on
    if st.lead_shaping and COAST_TTC > 0. and not reset_state and output_a_target > 0. and present:
      closing = v_ego - float(lead.vLead)
      if closing > COAST_CLOSING and (d_now - self.follow_distance) / closing < COAST_TTC:
        output_a_target = 0.
        self.limit = Limit.coast

    # a different car ahead: do not close on it before its speed is known
    if present and self.cut_prev_d > 0. and d_now - self.cut_prev_d > CUT_JUMP_M:
      self.cut_frames, self.cut_d0, self.cut_since = int(CUT_S / self.dt), d_now, 0
    else:
      self.cut_frames = max(self.cut_frames - 1, 0)
      self.cut_since += 1
    self.cut_prev_d = d_now
    self.cut_lost = 0 if present else self.cut_lost + 1
    if self.cut_lost * self.dt >= CUT_LOST_S:
      self.cut_frames = 0
    if st.lead_shaping and self.cut_frames > 0 and not reset_state and v_ego > CUT_MIN_V and output_a_target > CUT_A:
      away = (present and self.cut_since * self.dt >= CUT_AWAY_MIN_S
              and (d_now - self.cut_d0) / (self.cut_since * self.dt) > CUT_AWAY)
      if not away and not is_chase(lead):
        output_a_target = CUT_A
        self.limit = Limit.cut

    # let braking build up - unless it is urgent, or it is the model's or a stop's
    if (st.brake_onset and BRAKE_ONSET_J > 0. and not reset_state and not model_led and not should_stop and not fcw
        and output_a_target < min(a_prev, 0.0) and v_ego >= BRAKE_ONSET_MIN_V):
      closing = v_ego - float(lead.vLead) if present else 0.
      urgent = present and ((closing > 0.1 and d_now / closing < BRAKE_ONSET_TTC) or
                            (self.follow_distance > 0. and d_now < BRAKE_ONSET_GAP * self.follow_distance))
      held_back = min(a_prev, 0.0) - BRAKE_ONSET_J * self.dt
      if not urgent and held_back > output_a_target:
        output_a_target = held_back
        self.limit = Limit.brakeOnset

    return float(output_a_target)
