"""Stopped behind a car, stay stopped until that car has actually gone.

The latch from henryopen/openpilot's standstill_hold (2026-09-26), and only the latch. Once the car is
stopped behind a radar lead that is itself stopped, it remembers where the lead was and lets go only
when the lead has moved RELEASE_M away from that point on its own - measured as dRel plus the distance
we have covered, so it reads the same whether we moved or not. While it holds, the planner keeps the
accel at or below zero and keeps shouldStop set.

Left behind on purpose: the fork's crawl and close-in arms, its brake requests inside the stop distance
and its launch ramp. Those were built around how a Hyundai Custin answers a stop request and how it
surges when it pulls away; none of that has been measured on a Toyota. How this latch gets on with
Toyota's own standstill and resume has not been driven either, which is why it is off by default.
"""

ARM_SPEED = 0.3        # m/s
ARM_MAX_GAP = 9.0      # m
ARM_LEAD_STILL = 0.5   # m/s - the lead has to be stopped too; a crawling lead is followed, not held
RELEASE_M = 1.0        # m the lead moves away on its own before we go
LOST_GRACE = 1.0       # s - do not let go on a radar blink
DREL_TAU = 0.3         # s - smooth dRel before differencing; the range on a stopped car jitters
JUMP_M = 2.0           # m - a reading this far from the filter is some other return: counts as lead missing


class StandstillHold:
  def __init__(self, dt: float):
    self.dt = dt
    self.alpha = dt / (DREL_TAU + dt)
    # After a release, not again until we have actually moved off: a lead creeping away at under
    # ARM_LEAD_STILL would otherwise re-arm it on the very next frame and the car would never get going.
    self.need_move = False
    self.reset()

  def reset(self) -> None:
    self.active = False
    self.odo = 0.0
    self.d_f = 0.0
    self.ref_d = 0.0
    self.ref_odo = 0.0
    self.lost_t = 0.0
    self.moved = 0.0

  def _release(self) -> None:
    self.reset()
    self.need_move = True

  def update(self, enabled: bool, v_ego: float, lead, gas_pressed: bool) -> bool:
    """lead is radarState.leadOne. Returns True while the car should be held stopped."""
    self.odo += v_ego * self.dt
    radar_lead = bool(lead.present and lead.radar)

    if not enabled or gas_pressed:
      self.reset()
      self.need_move = False
      return False

    if v_ego >= ARM_SPEED:
      self.need_move = False

    if not self.active:
      if (radar_lead and not self.need_move and v_ego < ARM_SPEED
          and float(lead.vLead) < ARM_LEAD_STILL and float(lead.dRel) < ARM_MAX_GAP):
        self.active = True
        self.d_f = self.ref_d = float(lead.dRel)
        self.ref_odo = self.odo
        self.lost_t = 0.0
        self.moved = 0.0
      return self.active

    if radar_lead and abs(float(lead.dRel) - self.d_f) <= JUMP_M:
      self.lost_t = 0.0
      self.d_f += self.alpha * (float(lead.dRel) - self.d_f)
      self.moved = self.d_f + (self.odo - self.ref_odo) - self.ref_d
      if self.moved >= RELEASE_M:
        self._release()
    else:
      self.lost_t += self.dt
      if self.lost_t >= LOST_GRACE:
        self._release()
    return self.active
