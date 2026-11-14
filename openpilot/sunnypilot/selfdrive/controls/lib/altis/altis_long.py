"""Altis longitudinal tune.

A port of the driving-behaviour half of henryopen/openpilot (Custin openpilot, branch c4-only,
2026-11-06) onto sunnypilot's planner, for a 12th generation Toyota Corolla Altis (TSS 2.0).

What is here, all of it switched by the AltisLongTune param and only on a Toyota with
openpilot longitudinal control:

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
  - standstill_hold, lead_stop, long_deadzone, dash_to_true: fitted to the Custin's own actuator,
    stop request and speedometer. They describe that car, not this one.
  - lead_view and the lane-change follow gap: they need a modified MPC.
  - turn_slow, road_memory: they need the fork's own map daemon.
  - everything lateral and everything in opendbc/panda: Toyota's stock limits are left alone.

Every number below was measured on a Hyundai Custin by its one driver. They are a starting point
for an Altis, not a calibration. Nothing in this file can ask for more deceleration than the
planner already wanted, with one exception: A_CRUISE_MIN lets the cruise candidate slow to a
lowered set speed at up to 1.2 m/s^2 where stock stops at 0.5.
"""
import math

import numpy as np

from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.altis.curve_speed import CurveSpeedControl

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

    self.curve_speed = CurveSpeedControl()
    self.weak_lead_frames = 0
    self.follow_distance = 0.
    self.a_cruise_max = 0.
    self.curve_limited = False
    self.weak_lead = False

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

  def cruise_accel(self, sm, v_cruise: float, v_ego: float, a_cruise_prev: float, t_follow: float,
                   long_active: bool) -> float:
    """The cruise candidate. Replaces the planner's get_cruise_accel while the tune is active."""
    cs = sm['carState']
    lead = sm['radarState'].leadOne

    self.follow_distance = (self._safe_obstacle_distance(v_ego, t_follow)
                            - self._stopped_equivalence(float(lead.vLead))) if lead.present else 0.

    lead_free = self._lead_is_far(lead, v_ego, t_follow)
    max_accel = float(np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS_FREE if lead_free else A_CRUISE_MAX_VALS))
    if (lead.present and float(lead.vLead) > v_ego + CHASE_DV_PLAN
        and float(lead.dRel) >= self.follow_distance and is_chase(lead)):
      max_accel = max(max_accel, float(np.interp(v_ego, CHASE_BP, CHASE_VALS)))

    # not hard out of a corner: what is left of the budget once the sideways part is taken
    a_total_max = float(np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V))
    a_y = v_ego ** 2 * float(sm['controlsState'].curvature)
    max_accel = min(max_accel, math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.)))
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

    # ease off before a corner the model can see
    self.curve_speed.update(sm, long_active, cs.gasPressed, v_ego, cs.aEgo)
    self.curve_limited = self.curve_speed.is_active and self.curve_speed.a_target < a_cruise
    if self.curve_speed.is_active:
      a_cruise = min(a_cruise, float(self.curve_speed.a_target))

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
    self.weak_lead = self.weak_lead_frames >= WEAK_LEAD_MIN_FRAMES
    if self.weak_lead:
      a_cruise = min(a_cruise, 0.)

    return a_cruise

  def shape_output(self, sm, output_a_target: float, a_prev: float, v_ego: float, reset_state: bool,
                   model_led: bool, should_stop: bool, fcw: bool) -> float:
    """After the candidates are resolved. model_led: the driving model's own plan won the min().

    The first two only ever take acceleration away. The third lets braking arrive later, so it stands
    down for anything urgent and for anything the model or a stop asked for."""
    lead = sm['radarState'].leadOne
    present = bool(lead.present)
    d_now = float(lead.dRel) if present else -1.

    # no throttle towards a lead we are already closing on
    if COAST_TTC > 0. and not reset_state and output_a_target > 0. and present:
      closing = v_ego - float(lead.vLead)
      if closing > COAST_CLOSING and (d_now - self.follow_distance) / closing < COAST_TTC:
        output_a_target = 0.

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
    if self.cut_frames > 0 and not reset_state and v_ego > CUT_MIN_V and output_a_target > CUT_A:
      away = (present and self.cut_since * self.dt >= CUT_AWAY_MIN_S
              and (d_now - self.cut_d0) / (self.cut_since * self.dt) > CUT_AWAY)
      if not away and not is_chase(lead):
        output_a_target = CUT_A

    # let braking build up - unless it is urgent, or it is the model's or a stop's
    if (BRAKE_ONSET_J > 0. and not reset_state and not model_led and not should_stop and not fcw
        and output_a_target < min(a_prev, 0.0) and v_ego >= BRAKE_ONSET_MIN_V):
      closing = v_ego - float(lead.vLead) if present else 0.
      urgent = present and ((closing > 0.1 and d_now / closing < BRAKE_ONSET_TTC) or
                            (self.follow_distance > 0. and d_now < BRAKE_ONSET_GAP * self.follow_distance))
      if not urgent:
        output_a_target = max(output_a_target, min(a_prev, 0.0) - BRAKE_ONSET_J * self.dt)

    return float(output_a_target)
