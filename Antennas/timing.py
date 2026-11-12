"""Wall-clock timing of the time-domain and WaveHoltz runs on an otherwise idle machine.

The parent scripts' cache stores the TD run length but not its wall time, and the
WaveHoltz wall times in results/runs were measured while the solve_cw queue was
competing for the CPU.  This script re-runs, with a timer and nothing else on the
machine (pause run_cw_queue.sh first):

  td       every 2D configuration of the figures/tables (horn.py, yagi_uda.py,
           unchanged), into results/td_timed/ (the parent cache is not touched);
           also checks that the re-run reproduces the cached TD pattern;
  td_conv  the run-length study of td_convergence.py (results/td_convergence/);
  wh       every WaveHoltz configuration (same solver settings; overwrites the
           cached result, which is deterministic, and records the idle time).

Results go to results/timing.json, keyed like "horn_N16_f0.5_E0".

    python timing.py --td --td-conv --wh
"""

import argparse
import json
import time

import numpy as np
import meep as mp

import cw_common as W
from cw_common import C
import horn as H
import yagi_uda as Y
import horn_cw as HC
import yagi_cw as YC
import td_convergence as TC

OUT = W.RESULTS / "timing.json"
STEPS_PER_UNIT = {"horn": 6 / 0.5, "yagi": 6 / 0.5}   # resolution / Courant of the TD runs


def configs():
    horn = set()
    horn |= {(f, 0.0) for f in HC.BW_FREQS}
    horn |= {(f, d) for f in (0.3, 1.5, 3.0) for d in (0.0, 10.0, 20.0)}
    horn |= {(f, d) for f in (0.5, 1.0, 2.0, 4.0) for d in (0.0, 80.0)}
    horn |= {(f, d) for f in (0.5, 5.0) for d in (0.0, 10.0, 20.0, 30.0)}
    yagi = set()
    yagi |= {(n, 5.0, d) for n in (8, 16) for d in np.arange(0, 81, 10.0)}
    yagi |= {(16, f, d) for f in (2.5, 5.0) for d in (0.0, 20.0, 40.0, 60.0)}
    return ([("horn", 16, f, float(d)) for f, d in sorted(horn)] +
            [("yagi", n, f, float(d)) for n, f, d in sorted(yagi)])


def key(ant, n, f, d, periods=None):
    k = f"{ant}_N{n}_f{f:g}_E{d:g}"
    return k + (f"_P{periods}" if periods else "")


def td_ns(ant, outdir, until_ns=6.67, min_periods=10):
    a = TC.td_args(min_periods, force=True)
    a.outdir, a.until_ns = outdir, until_ns
    if ant == "yagi":
        a.pad, a.sx, a.wire, a.backplane_x = 4.0, 42.0, 0.2, None
    return a


def timed(fn):
    t0 = time.time()
    r = fn()
    return r, time.time() - t0


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(db):
    if mp.am_master():
        OUT.write_text(json.dumps(db, indent=1, sort_keys=True))


def run_td(db):
    db.setdefault("td", {})
    outdir = W.RESULTS / "td_timed"
    for ant, n, f, d in configs():
        k = key(ant, n, f, d)
        a = td_ns(ant, outdir)
        if ant == "horn":
            r, wall = timed(lambda: H.simulate(a, f, d))
        else:
            r, wall = timed(lambda: Y.simulate(a, n, f, d, 0.0, 2))
        until = r["meta"]["until"]
        ref = W.td_result(antenna=ant, dim=2, n=n, f_ghz=f, dphi_e=d, dphi_h=0.0,
                          resolution=6, dpml=1.0)
        same = HC.pattern_diff_db(r, ref) if ref is not None else None
        db["td"][k] = dict(until=until, periods=until * C.ghz(f),
                           steps=int(round(until * STEPS_PER_UNIT[ant])), wall_s=wall,
                           reproduces_cached_dB=same)
        if mp.am_master():
            print(f"td {k}: {until * C.ghz(f):.1f} periods, {wall:.1f} s, "
                  f"vs cached {same}")
        save(db)


def run_td_conv(db):
    db.setdefault("td_conv", {})
    for f, d in TC.CASES:
        for p in TC.PERIODS:
            a = TC.td_args(p, force=True)
            r, wall = timed(lambda: H.simulate(a, f, d))
            until = r["meta"]["until"]
            db["td_conv"][key("horn", 16, f, d, p)] = dict(
                until=until, periods=p, steps=int(round(until * STEPS_PER_UNIT["horn"])),
                wall_s=wall)
            if mp.am_master():
                print(f"td_conv f={f} dPhi={d} {p} periods: {wall:.1f} s")
            save(db)


def run_wh(db):
    db.setdefault("wh", {})
    ah, _ = HC.parse_args(["run", "--solvers", "wh", "--show", "", "--force"])
    ay, _ = YC.parse_args(["run", "--solvers", "wh", "--show", "", "--force"])
    for ant, n, f, d in configs():
        if ant == "horn":
            r, wall = timed(lambda: HC.simulate(ah, "wh", f, d))
        else:
            r, wall = timed(lambda: YC.simulate(ay, "wh", n, f, d))
        st = W.stats_of(r)
        db["wh"][key(ant, n, f, d)] = dict(iters=st["iters"], steps=st["timesteps"],
                                          solve_s=st["wall_s"], wall_s=wall)
        if mp.am_master():
            print(f"wh {key(ant, n, f, d)}: {st['iters']} it, solve {st['wall_s']:.1f} s, "
                  f"total {wall:.1f} s")
        save(db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--td", action="store_true")
    ap.add_argument("--td-conv", action="store_true")
    ap.add_argument("--wh", action="store_true")
    opt = ap.parse_args()
    db = load()
    if opt.td:
        run_td(db)
    if opt.td_conv:
        run_td_conv(db)
    if opt.wh:
        run_wh(db)


if __name__ == "__main__":
    main()
