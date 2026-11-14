from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.altis import altis_long
from openpilot.sunnypilot.selfdrive.controls.lib.altis.altis_long import AltisLongTune

COMFORT_BRAKE, STOP_DISTANCE, T_FOLLOW = 2.5, 6.0, 1.45
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])


def make_cp(brand="toyota", op_long=True):
  return SimpleNamespace(brand=brand, openpilotLongitudinalControl=op_long)


def make_lead(present=False, d_rel=0., v_lead=0., v_rel=0., a_lead=0., radar=True):
  return SimpleNamespace(present=present, dRel=d_rel, vLead=v_lead, vRel=v_rel, aLeadK=a_lead, radar=radar)


def make_sm(v_ego=10., lead=None, curvature=0., plan_curvature=0., leads_v3=(), gas=False):
  plan_v = np.full(33, max(v_ego, 1.))
  model = SimpleNamespace(
    position=SimpleNamespace(x=(plan_v * T_IDXS).tolist()),
    velocity=SimpleNamespace(x=plan_v.tolist()),
    orientationRate=SimpleNamespace(z=(plan_v * plan_curvature).tolist()),
    meta=SimpleNamespace(laneChangeState=log.LaneChangeState.off),
    leadsV3=list(leads_v3))
  return {'carState': SimpleNamespace(vEgo=v_ego, aEgo=0., gasPressed=gas),
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
    a = tune.cruise_accel(sm, v_cruise, v_ego, a, T_FOLLOW, True)
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
    a = tune.cruise_accel(sm, 30., 10., 0., T_FOLLOW, True)
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
      a = tune.cruise_accel(sm, 30., 15., a, T_FOLLOW, True)
      assert tune.weak_lead == (i + 1 >= altis_long.WEAK_LEAD_MIN_FRAMES)
    assert a <= 0.
    tune.cruise_accel(make_sm(v_ego=15.), 30., 15., a, T_FOLLOW, True)
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
    tune.cruise_accel(sm, 30., v_ego, 0., T_FOLLOW, True)
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
      tune.cruise_accel(sm, 30., v_ego, 0., T_FOLLOW, True)
      a_in = float(rng.uniform(-3.5, 2.0))
      a_out = tune.shape_output(sm, a_in, a_prev, v_ego, False, bool(rng.random() < 0.3), bool(rng.random() < 0.1), False)
      if a_in >= 0.:
        assert 0. <= a_out <= a_in
      else:
        assert a_in <= a_out <= min(a_prev, 0.) - altis_long.BRAKE_ONSET_J * DT_MDL + 1e-9 or a_out == a_in
      a_prev = a_out
