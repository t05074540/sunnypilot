#!/usr/bin/env python3
"""Measure this car and this driver, to replace the Custin numbers in the Altis tune.

  python3 openpilot/sunnypilot/tools/altis_accel_report.py <route or rlog> [<route or rlog> ...]

Every value in altis_long.py and altis/curve_speed.py was measured on a Hyundai Custin. This reads
drives of the car the tune runs on and prints the same tables those values were set from:

  1. acceleration by speed band - the driver's own foot against what openpilot did, with the road
     clear and behind a lead, and how often openpilot sat on the tune's ceiling
     -> A_CRUISE_MAX_VALS, A_CRUISE_MAX_VALS_FREE, ACCEL_PROFILE_SCALE
  2. how much of the requested acceleration the car delivers -> whether the ceilings read as felt
  3. lateral acceleration in corners by speed band, the driver against openpilot
     -> curve_speed's _V2_LAT_V, CURVE_COMFORT_SCALE
  4. how often openpilot accelerates and then brakes within 2 s -> lead shaping
  5. what the tune's limiter was, as a share of engaged time (needs the tune on while driving)

It suggests ceilings from the driver's own acceleration; it changes nothing. Use several drives:
the high-speed bands in particular are thin on one.
"""
import argparse
from collections import defaultdict

import numpy as np

from openpilot.common.constants import CV
from openpilot.tools.lib.logreader import LogReader
from openpilot.sunnypilot.selfdrive.controls.lib.altis.altis_long import A_CRUISE_MAX_BP, LIMIT_NAMES

BANDS_KPH = [0, 10, 18, 27, 36, 45, 54, 72, 90, 110, 140]
LAT_BANDS_KPH = [10, 20, 40, 60, 80, 100, 140]
CLEAR_M = 40.          # no lead within this: the road is clear
MIN_SAMPLES = 100      # frames at 20 Hz before a band is believed
ACCEL_MIN = 0.1        # m/s^2: accelerating
CORNER_LAT = 0.8       # m/s^2: in a corner
THEN_BRAKE_S = 2.0


def band(v_kph, bands):
  i = int(np.searchsorted(bands, v_kph, side='right')) - 1
  return i if 0 <= i < len(bands) - 1 else None


def pct(values, q):
  return float(np.percentile(values, q)) if len(values) >= MIN_SAMPLES else float('nan')


def read(route):
  """One row per carState, with the latest of everything else."""
  rows = []
  last = dict(enabled=False, a_target=0., lead=False, d_rel=0., curvature=0., limit=0, altis=False, a_max=0.)
  for msg in LogReader(route):
    w = msg.which()
    if w == 'selfdriveState':
      last['enabled'] = msg.selfdriveState.enabled
    elif w == 'longitudinalPlan':
      last['a_target'] = msg.longitudinalPlan.aTarget
    elif w == 'longitudinalPlanSP':
      altis = msg.longitudinalPlanSP.altis
      last['altis'], last['limit'], last['a_max'] = altis.active, altis.limit.raw, altis.aCruiseMax
    elif w == 'radarState':
      lead = msg.radarState.leadOne
      last['lead'], last['d_rel'] = lead.present, lead.dRel
    elif w == 'controlsState':
      last['curvature'] = msg.controlsState.curvature
    elif w == 'carState':
      cs = msg.carState
      rows.append((msg.logMonoTime * 1e-9, cs.vEgo, cs.aEgo, cs.gasPressed, cs.brakePressed, last['enabled'], last['a_target'],
                   last['lead'], last['d_rel'], last['curvature'], last['limit'], last['altis'], last['a_max']))
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('routes', nargs='+')
  args = parser.parse_args()

  rows = []
  for route in args.routes:
    got = read(route)
    print(f"{route}: {len(got)} carState frames")
    rows += got
  if not rows:
    return
  t, v, a, gas, brake, enabled, a_target, lead, d_rel, curvature, limit, altis, a_max = (np.array(c) for c in zip(*rows, strict=True))
  v_kph = v * CV.MS_TO_KPH
  dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
  clear = ~lead.astype(bool) | (d_rel > CLEAR_M)
  op = enabled.astype(bool) & ~gas.astype(bool) & ~brake.astype(bool)
  him = gas.astype(bool) & ~brake.astype(bool)

  print("\n1. Acceleration while accelerating, median / P75 (m/s^2). 'on ceiling': openpilot's request within 8% of the tune's ceiling.")
  print(f"{'km/h':>10} | {'openpilot, clear':>18} {'driver, clear':>16} | {'openpilot, lead':>16} {'driver, lead':>14} | {'on ceiling':>10}")
  suggest = {}
  for i in range(len(BANDS_KPH) - 1):
    sel = (v_kph >= BANDS_KPH[i]) & (v_kph < BANDS_KPH[i + 1]) & (a > ACCEL_MIN)
    cells = []
    for who, road in ((op, clear), (him, clear), (op, ~clear), (him, ~clear)):
      x = a[sel & who & road]
      cells.append(f"{pct(x, 50):5.2f} / {pct(x, 75):4.2f}")
    on = sel & op & altis.astype(bool) & (a_max > 0.)
    share = float(np.mean(a_target[on] >= 0.92 * a_max[on])) * 100 if on.sum() >= MIN_SAMPLES else float('nan')
    print(f"{BANDS_KPH[i]:>4}-{BANDS_KPH[i + 1]:<5} | {cells[0]:>18} {cells[1]:>16} | {cells[2]:>16} {cells[3]:>14} | {share:9.0f}%")
    suggest[i] = (pct(a[sel & him & clear], 50), pct(a[sel & him & clear], 75), pct(a[sel & him & ~clear], 50))

  print("\n   The driver's own, at the tune's breakpoints (interpolated between the bands above; nan = not enough data).")
  print("   The Custin's curves were set between the driver's median and P75 with the road clear, and near the median behind a lead.")
  mids = [(BANDS_KPH[i] + BANDS_KPH[i + 1]) / 2 * CV.KPH_TO_MS for i in suggest]
  for name, col in (("clear, median", 0), ("clear, P75", 1), ("lead, median", 2)):
    vals = np.array([suggest[i][col] for i in suggest])
    ok = ~np.isnan(vals)
    line = np.interp(A_CRUISE_MAX_BP, np.array(mids)[ok], vals[ok], left=np.nan, right=np.nan) if ok.sum() >= 2 else np.full(len(A_CRUISE_MAX_BP), np.nan)
    print(f"   {name:>14}: [" + ", ".join(f"{x:.2f}" for x in line) + "]")

  print("\n2. What the car delivers of what is asked (openpilot accelerating, request over 0.2): median aEgo / aTarget.")
  for i in range(len(BANDS_KPH) - 1):
    sel = (v_kph >= BANDS_KPH[i]) & (v_kph < BANDS_KPH[i + 1]) & op & (a_target > 0.2)
    if sel.sum() >= MIN_SAMPLES:
      print(f"   {BANDS_KPH[i]:>4}-{BANDS_KPH[i + 1]:<5} {np.median(a[sel] / a_target[sel]):5.2f}   ({int(sel.sum())} frames)")

  print(f"\n3. Lateral acceleration in corners (over {CORNER_LAT} m/s^2), median / P90. curve_speed's comfort curve is set from the driver's.")
  lat = v ** 2 * np.abs(curvature)
  for i in range(len(LAT_BANDS_KPH) - 1):
    sel = (v_kph >= LAT_BANDS_KPH[i]) & (v_kph < LAT_BANDS_KPH[i + 1]) & (lat > CORNER_LAT)
    x_op, x_him = lat[sel & enabled.astype(bool)], lat[sel & ~enabled.astype(bool)]
    print(f"   {LAT_BANDS_KPH[i]:>4}-{LAT_BANDS_KPH[i + 1]:<5} openpilot {pct(x_op, 50):5.2f} / {pct(x_op, 90):4.2f}"
          f"    driver {pct(x_him, 50):5.2f} / {pct(x_him, 90):4.2f}")

  print(f"\n4. Accelerating past +0.2 and then braking past -0.3 within {THEN_BRAKE_S:.0f} s, times a minute.")
  n = int(THEN_BRAKE_S / dt)
  for name, who in (("openpilot", enabled.astype(bool)), ("driver", ~enabled.astype(bool))):
    idx = np.flatnonzero(who & (a > 0.2) & (v > 3.))
    count, last_hit = 0, -n
    for j in idx:
      if j - last_hit >= n and np.any(a[j:j + n] < -0.3):
        count, last_hit = count + 1, j
    minutes = who.sum() * dt / 60.
    print(f"   {name:>10}: {count / minutes if minutes > 1. else float('nan'):5.2f}   ({minutes:.0f} min)")

  print("\n5. The tune's limiter, share of engaged time.")
  on = altis.astype(bool) & enabled.astype(bool)
  if on.sum() < MIN_SAMPLES:
    print("   The tune was not active on these drives.")
  else:
    counts = defaultdict(int)
    for x in limit[on]:
      counts[int(x)] += 1
    for k, c in sorted(counts.items(), key=lambda kv: -kv[1]):
      print(f"   {LIMIT_NAMES[k] if k < len(LIMIT_NAMES) else k:>10}: {100 * c / on.sum():5.1f}%")


if __name__ == "__main__":
  main()
