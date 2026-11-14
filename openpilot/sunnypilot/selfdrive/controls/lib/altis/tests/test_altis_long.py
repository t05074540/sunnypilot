from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.altis import altis_long
from openpilot.sunnypilot.selfdrive.controls.lib.altis.altis_long import AltisLongTune

COMFORT_BRAKE, STOP_DISTANCE, T_FOLLOW = 2.5, 6.0, 1.45
STOCK = 0.33   # stands in for the planner's own cruise accel
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])


def make_cp(brand="toyota", op_long=True):
  return SimpleNamespace(brand=brand, openpilotLongitudinalControl=op_long)


def make_lead(present=False, d_rel=0., v_lead=0., v_rel=0., a_lead=0., radar=True):
  return SimpleNamespace(present=present, dRel=d_rel, vLead=v_lead, vRel=v_rel, aLeadK=a_lead, radar=radar)


def make_sm(v_ego=10., lead=None, curvature=0., plan_curvature=0., leads_v3=(), gas=False, brake=False,
            lc_state=log.LaneChangeState.off, lc_dir=log.LaneChangeDirection.none, left_bs=False, right_bs=False):
  plan_v = np.full(33, max(v_ego, 1.))
  model = SimpleNamespace(
    position=SimpleNamespace(x=(plan_v * T_IDXS).tolist()),
    velocity=SimpleNamespace(x=plan_v.tolist()),
    orientationRate=SimpleNamespace(z=(plan_v * plan_curvature).tolist()),
    meta=SimpleNamespace(laneChangeState=lc_state, laneChangeDirection=lc_dir),
    leadsV3=list(leads_v3))
  return {'carState': SimpleNamespace(vEgo=v_ego, aEgo=0., gasPressed=gas, brakePressed=brake,
                                      leftBlindspot=left_bs, rightBlindspot=right_bs),
          'radarState': SimpleNamespace(leadOne=lead or make_lead()),
          'controlsState': SimpleNamespace(curvature=curvature),
          'modelV2': model}


def make_tune(**kw):
  tune = AltisLongTune(make_cp(**kw), COMFORT_BRAKE, STOP_DISTANCE)
  tune.enabled = True
  return tune


def settle_cruise(tune, sm, v_cruise, v_ego, frames=200):
  a = 0.
  for _ in range(frames):
    a = tune.cruise_accel(sm, v_cruise, v_ego, a, T_FOLLOW, True, STOCK)
  return a


class TestActive:
  def test_needs_param(self):
    tune = AltisLongTune(make_cp(), COMFORT_BRAKE, STOP_DISTANCE)
    assert not tune.active
    tune.enabled = True
    assert tune.active

  def test_toyota_with_op_long_only(self):
    assert not make_tune(brand="hyundai").active
    assert not make_tune(op_long=False).active


class TestCruiseAccel:
  def test_clear_road_ceiling(self):
    a = settle_cruise(make_tune(), make_sm(v_ego=10.), v_cruise=30., v_ego=10.)
    assert abs(a - np.interp(10., altis_long.A_CRUISE_MAX_BP, altis_long.A_CRUISE_MAX_VALS_FREE)) < 1e-6

  def test_lead_in_reach_lowers_ceiling(self):
    lead = make_lead(present=True, d_rel=20., v_lead=10.)
    a = settle_cruise(make_tune(), make_sm(v_ego=10., lead=lead), v_cruise=30., v_ego=10.)
    assert abs(a - np.interp(10., altis_long.A_CRUISE_MAX_BP, altis_long.A_CRUISE_MAX_VALS)) < 1e-6

  def test_chase_raises_ceiling_for_a_leaving_radar_lead_only(self):
    for radar, expect in ((True, altis_long.CHASE_VALS), (False, altis_long.A_CRUISE_MAX_VALS_FREE)):
      lead = make_lead(present=True, d_rel=80., v_lead=13., v_rel=3., a_lead=0.6, radar=radar)
      a = settle_cruise(make_tune(), make_sm(v_ego=10., lead=lead), v_cruise=30., v_ego=10.)
      assert abs(a - np.interp(10., altis_long.CHASE_BP, expect)) < 1e-6

  def test_never_past_stock_accel_ceiling(self):
    for v in np.arange(0., 40., 0.5):
      assert settle_cruise(make_tune(), make_sm(v_ego=v), v_cruise=45., v_ego=v) <= 1.2 + 1e-6

  def test_deadzone_holds_zero(self):
    assert settle_cruise(make_tune(), make_sm(v_ego=20.), v_cruise=20.2, v_ego=20.) == 0.

  def test_tapers_into_set_speed(self):
    far = settle_cruise(make_tune(), make_sm(v_ego=14.), v_cruise=20., v_ego=14.)
    near = settle_cruise(make_tune(), make_sm(v_ego=19.5), v_cruise=20., v_ego=19.5)
    assert 0. < near < far

  def test_slowing_to_a_lower_set_speed_is_bounded(self):
    a = settle_cruise(make_tune(), make_sm(v_ego=30.), v_cruise=10., v_ego=30.)
    assert a == altis_long.A_CRUISE_MIN

  def test_jerk_limited(self):
    tune, sm = make_tune(), make_sm(v_ego=10.)
    a = tune.cruise_accel(sm, 30., 10., 0., T_FOLLOW, True, STOCK)
    assert 0. < a <= np.interp(10., altis_long.J_CRUISE_BP, altis_long.J_CRUISE_VALS) * DT_MDL + 1e-9

  def test_no_hard_acceleration_while_cornering(self):
    straight = settle_cruise(make_tune(), make_sm(v_ego=20.), v_cruise=40., v_ego=20.)
    cornering = settle_cruise(make_tune(), make_sm(v_ego=20., curvature=1.69 / 20. ** 2), v_cruise=40., v_ego=20.)
    assert cornering < straight

  def test_weak_lead_needs_to_last(self):
    weak = [SimpleNamespace(x=[70.], prob=0.35)]
    tune, sm = make_tune(), make_sm(v_ego=15., leads_v3=weak)
    a = 0.
    for i in range(altis_long.WEAK_LEAD_MIN_FRAMES + 5):
      a = tune.cruise_accel(sm, 30., 15., a, T_FOLLOW, True, STOCK)
      assert tune.weak_lead == (i + 1 >= altis_long.WEAK_LEAD_MIN_FRAMES)
    assert a <= 0.
    tune.cruise_accel(make_sm(v_ego=15.), 30., 15., a, T_FOLLOW, True, STOCK)
    assert not tune.weak_lead

  def test_slows_for_a_corner_ahead(self):
    tune = make_tune()
    sm = make_sm(v_ego=25., plan_curvature=0.02)   # 12.5 m/s^2 at this speed: far past any comfort curve
    a = settle_cruise(tune, sm, v_cruise=25., v_ego=25., frames=100)
    assert tune.curve_limited and a < -0.3

  def test_straight_road_is_not_a_corner(self):
    tune = make_tune()
    settle_cruise(tune, make_sm(v_ego=25.), v_cruise=30., v_ego=25., frames=100)
    assert not tune.curve_limited


class TestShapeOutput:
  def shape(self, tune, sm, a, a_prev=0., v_ego=15., reset=False, model_led=False, should_stop=False, fcw=False):
    return tune.shape_output(sm, a, a_prev, v_ego, reset, model_led, should_stop, fcw)

  def with_follow(self, lead, v_ego=15.):
    tune, sm = make_tune(), make_sm(v_ego=v_ego, lead=lead)
    tune.cruise_accel(sm, 30., v_ego, 0., T_FOLLOW, True, STOCK)
    return tune, sm

  def test_coast_towards_a_slower_lead(self):
    tune, sm = self.with_follow(make_lead(present=True, d_rel=60., v_lead=12., v_rel=-3.))
    assert self.shape(tune, sm, 0.5) == 0.

  def test_follows_a_faster_lead(self):
    tune, sm = self.with_follow(make_lead(present=True, d_rel=60., v_lead=17., v_rel=2.))
    assert self.shape(tune, sm, 0.5) == 0.5

  def test_cut_holds_acceleration_for_a_new_lead(self):
    tune, sm = self.with_follow(make_lead(present=True, d_rel=30., v_lead=15.))
    self.shape(tune, sm, 0.5)
    tune, sm2 = tune, make_sm(v_ego=15., lead=make_lead(present=True, d_rel=45., v_lead=15.2, v_rel=0.2))
    assert self.shape(tune, sm2, 0.8) == altis_long.CUT_A
    for _ in range(int(altis_long.CUT_S / DT_MDL) + 1):
      a = self.shape(tune, sm2, 0.8)
    assert a == 0.8

  def test_cut_ends_when_no_lead_is_left(self):
    tune, sm = self.with_follow(make_lead(present=True, d_rel=30., v_lead=15.))
    self.shape(tune, sm, 0.5)
    self.shape(tune, make_sm(v_ego=15., lead=make_lead(present=True, d_rel=45., v_lead=15.)), 0.8)
    empty = make_sm(v_ego=15.)
    for _ in range(int(altis_long.CUT_LOST_S / DT_MDL) + 1):
      a = self.shape(tune, empty, 0.8)
    assert a == 0.8

  def test_braking_builds_up(self):
    tune, sm = self.with_follow(make_lead(present=True, d_rel=80., v_lead=14.5, v_rel=-0.5))
    assert abs(self.shape(tune, sm, -1.0, a_prev=0.) + altis_long.BRAKE_ONSET_J * DT_MDL) < 1e-9

  def test_urgent_braking_is_not_held_back(self):
    close = make_lead(present=True, d_rel=12., v_lead=8., v_rel=-7.)
    tune, sm = self.with_follow(close)
    assert self.shape(tune, sm, -2.5, a_prev=0.) == -2.5

  def test_model_stop_and_fcw_are_not_held_back(self):
    for kw in ({'model_led': True}, {'should_stop': True}, {'fcw': True}):
      tune, sm = self.with_follow(make_lead())
      assert self.shape(tune, sm, -2.0, a_prev=0., **kw) == -2.0

  def test_never_adds_acceleration_and_only_delays_braking_by_the_onset(self):
    rng = np.random.default_rng(0)
    tune = make_tune()
    a_prev = 0.
    for _ in range(5000):
      v_ego = float(rng.uniform(0., 35.))
      lead = make_lead(present=bool(rng.random() < 0.7), d_rel=float(rng.uniform(2., 120.)),
                       v_lead=float(rng.uniform(0., 35.)), a_lead=float(rng.uniform(-3., 1.)), radar=bool(rng.random() < 0.5))
      lead.vRel = lead.vLead - v_ego
      sm = make_sm(v_ego=v_ego, lead=lead)
      tune.cruise_accel(sm, 30., v_ego, 0., T_FOLLOW, True, STOCK)
      a_in = float(rng.uniform(-3.5, 2.0))
      a_out = tune.shape_output(sm, a_in, a_prev, v_ego, False, bool(rng.random() < 0.3), bool(rng.random() < 0.1), False)
      if a_in >= 0.:
        assert 0. <= a_out <= a_in
      else:
        assert a_in <= a_out <= min(a_prev, 0.) - altis_long.BRAKE_ONSET_J * DT_MDL + 1e-9 or a_out == a_in
      a_prev = a_out


class TestSettings:
  def test_accel_profile_scales_the_ceiling(self):
    got = []
    for profile in (0, 1, 2):
      tune = make_tune()
      tune.settings.accel_profile = profile
      got.append(settle_cruise(tune, make_sm(v_ego=10.), v_cruise=30., v_ego=10.))
    base = np.interp(10., altis_long.A_CRUISE_MAX_BP, altis_long.A_CRUISE_MAX_VALS_FREE)
    assert np.allclose(got, [s * base for s in altis_long.ACCEL_PROFILE_SCALE])

  def test_accel_curve_off_hands_back_the_planners_own(self):
    tune = make_tune()
    tune.settings.accel_curve = False
    assert settle_cruise(tune, make_sm(v_ego=10.), v_cruise=30., v_ego=10.) == STOCK

  def test_curve_speed_off(self):
    tune = make_tune()
    tune.settings.curve_speed = False
    settle_cruise(tune, make_sm(v_ego=25., plan_curvature=0.02), v_cruise=25., v_ego=25., frames=100)
    assert not tune.curve_limited

  def test_curve_comfort_changes_the_corner_speed(self):
    v_target = []
    for comfort in (0, 1, 2):
      tune = make_tune()
      tune.settings.curve_comfort = comfort
      settle_cruise(tune, make_sm(v_ego=25., plan_curvature=0.01), v_cruise=25., v_ego=25., frames=100)
      v_target.append(tune.curve_speed.v_target)
    assert v_target[0] < v_target[1] < v_target[2]

  def test_lead_shaping_off(self):
    lead = make_lead(present=True, d_rel=60., v_lead=12., v_rel=-3.)
    tune, sm = make_tune(), make_sm(v_ego=15., lead=lead)
    tune.settings.lead_shaping = False
    tune.cruise_accel(sm, 30., 15., 0., T_FOLLOW, True, STOCK)
    assert tune.shape_output(sm, 0.5, 0., 15., False, False, False, False) == 0.5

  def test_brake_onset_off(self):
    lead = make_lead(present=True, d_rel=80., v_lead=14.5, v_rel=-0.5)
    tune, sm = make_tune(), make_sm(v_ego=15., lead=lead)
    tune.settings.brake_onset = False
    tune.cruise_accel(sm, 30., 15., 0., T_FOLLOW, True, STOCK)
    assert tune.shape_output(sm, -1.0, 0., 15., False, False, False, False) == -1.0


class TestLimit:
  def test_reports_the_ceiling_then_the_corner(self):
    tune = make_tune()
    settle_cruise(tune, make_sm(v_ego=10.), v_cruise=30., v_ego=10.)
    assert tune.limit == altis_long.Limit.free
    settle_cruise(tune, make_sm(v_ego=25., plan_curvature=0.02), v_cruise=25., v_ego=25., frames=100)
    assert tune.limit == altis_long.Limit.curve

  def test_nothing_to_report_at_the_set_speed(self):
    tune = make_tune()
    settle_cruise(tune, make_sm(v_ego=20.), v_cruise=20., v_ego=20.)
    assert tune.limit == altis_long.Limit.none

  def test_names_match(self):
    names = [n for n in vars(altis_long.Limit) if not n.startswith('_')]
    assert sorted(names, key=lambda n: getattr(altis_long.Limit, n)) == list(altis_long.LIMIT_NAMES)


class TestLaneChangeGap:
  BASE = 1.45

  def run(self, tune, sm, frames, v_ego=25.):
    for _ in range(frames):
      t = tune.lane_change_t_follow(sm, self.BASE, v_ego, False)
    return t

  def test_off_by_default(self):
    sm = make_sm(v_ego=25., lc_state=log.LaneChangeState.laneChangeStarting)
    assert self.run(make_tune(), sm, 100) == self.BASE

  def test_ramps_down_and_back(self):
    tune = make_tune()
    tune.settings.lane_change_gap = True
    changing = make_sm(v_ego=25., lc_state=log.LaneChangeState.laneChangeStarting)
    assert abs(self.run(tune, changing, 1) - (self.BASE - altis_long.LC_RAMP_IN * DT_MDL)) < 1e-9
    assert self.run(tune, changing, 100) == altis_long.LC_T_FOLLOW
    assert altis_long.LC_T_FOLLOW < self.run(tune, make_sm(v_ego=25.), 20) < self.BASE
    assert self.run(tune, make_sm(v_ego=25.), 200) == self.BASE

  def test_never_longer_than_the_personality(self):
    tune = make_tune()
    tune.settings.lane_change_gap = True
    changing = make_sm(v_ego=25., lc_state=log.LaneChangeState.laneChangeStarting)
    for _ in range(100):
      assert tune.lane_change_t_follow(changing, 0.9, 25., False) == 0.9

  def test_back_at_once_when_it_matters(self):
    aborts = [dict(brake=True), dict(gas=True), dict(lc_dir=log.LaneChangeDirection.left, left_bs=True),
              dict(lc_dir=log.LaneChangeDirection.right, right_bs=True),
              dict(lead=make_lead(present=True, d_rel=30., v_lead=20., a_lead=-1.0))]
    for kw in aborts:
      tune = make_tune()
      tune.settings.lane_change_gap = True
      kw.setdefault('lc_dir', log.LaneChangeDirection.left)
      self.run(tune, make_sm(v_ego=25., lc_state=log.LaneChangeState.laneChangeStarting, lc_dir=kw['lc_dir']), 100)
      assert self.run(tune, make_sm(v_ego=25., lc_state=log.LaneChangeState.laneChangeStarting, **kw), 1) == self.BASE

  def test_not_at_low_speed(self):
    tune = make_tune()
    tune.settings.lane_change_gap = True
    sm = make_sm(v_ego=5., lc_state=log.LaneChangeState.laneChangeStarting)
    assert self.run(tune, sm, 100, v_ego=5.) == self.BASE


class TestStandstillHold:
  def hold_tune(self):
    tune = make_tune()
    tune.settings.standstill_hold = True
    return tune

  def step(self, tune, lead, v_ego=0., a=0.5, gas=False, reset=False):
    return tune.shape_output(make_sm(v_ego=v_ego, lead=lead, gas=gas), a, 0., v_ego, reset, False, False, False)

  def test_off_by_default(self):
    tune = make_tune()
    assert self.step(tune, make_lead(present=True, d_rel=4., v_lead=0.)) == 0.5 and not tune.hold_active

  def test_holds_behind_a_stopped_lead_through_radar_jitter(self):
    tune = self.hold_tune()
    rng = np.random.default_rng(1)
    for _ in range(200):
      lead = make_lead(present=True, d_rel=4. + float(rng.uniform(-0.3, 0.3)), v_lead=float(rng.uniform(0., 0.4)))
      assert self.step(tune, lead) == 0. and tune.hold_active

  def test_releases_once_the_lead_has_moved_a_metre(self):
    tune = self.hold_tune()
    self.step(tune, make_lead(present=True, d_rel=4., v_lead=0.))
    d, released_at = 4., None
    for _ in range(200):
      d += 1.0 * DT_MDL
      if self.step(tune, make_lead(present=True, d_rel=d, v_lead=1.0)) > 0.:
        released_at = d
        break
    assert released_at is not None and 5.0 <= released_at < 5.6

  def test_does_not_rearm_until_the_car_has_moved(self):
    tune = self.hold_tune()
    self.step(tune, make_lead(present=True, d_rel=4., v_lead=0.))
    for _ in range(60):
      self.step(tune, make_lead(present=True, d_rel=6., v_lead=0.4))
    assert not tune.hold_active
    assert self.step(tune, make_lead(present=True, d_rel=6., v_lead=0.2)) == 0.5

  def test_a_single_wild_reading_does_not_release(self):
    tune = self.hold_tune()
    still = make_lead(present=True, d_rel=4., v_lead=0.)
    self.step(tune, still)
    self.step(tune, make_lead(present=True, d_rel=43.7, v_lead=0.))
    assert self.step(tune, still) == 0. and tune.hold_active

  def test_lost_lead_releases_after_the_grace(self):
    tune = self.hold_tune()
    self.step(tune, make_lead(present=True, d_rel=4., v_lead=0.))
    for _ in range(int(1.0 / DT_MDL) + 2):
      a = self.step(tune, make_lead())
    assert a == 0.5 and not tune.hold_active

  def test_accelerator_and_disengage_release_at_once(self):
    for kw in (dict(gas=True), dict(reset=True)):
      tune = self.hold_tune()
      still = make_lead(present=True, d_rel=4., v_lead=0.)
      self.step(tune, still)
      assert self.step(tune, still, **kw) == 0.5 and not tune.hold_active

  def test_vision_only_lead_and_moving_car_do_not_arm(self):
    tune = self.hold_tune()
    assert self.step(tune, make_lead(present=True, d_rel=4., v_lead=0., radar=False)) == 0.5
    self.step(tune, make_lead(present=True, d_rel=4., v_lead=0.), v_ego=2.)   # coast takes the throttle, the hold does not arm
    assert not tune.hold_active

  def test_never_asks_for_more_braking(self):
    tune = self.hold_tune()
    assert self.step(tune, make_lead(present=True, d_rel=4., v_lead=0.), a=-0.7) == -0.7
