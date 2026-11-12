"""Time-domain solution with the start-up transient removed to a set tolerance.

The TD runs of the parent scripts accumulate the near-to-far DFT from t = 0, so
the start-up transient stays in the result and its weight falls only like 1/T
(td_convergence.py): ~4 dB in the side lobes at 10 periods, 0.2-0.3 dB at 160.
A 1e-6 error that way would need ~1e7 periods.  Here the transient is instead
waited out, and only the steady state is transformed:

1. Same geometry and sources as horn.py, real fields, Courant number chosen
   as for WaveHoltz (one window = `periods` periods = Nt whole steps), so the
   time-stepping steady state is the same discrete periodic solution that
   solve_wh computes.
2. A DFT of (Ey, Hz) over the whole interior (inside the PML) is accumulated
   from t = 0.  MEEP cannot reset a DFT, so after each window the running total is
   differenced against the previous one: w_k is the phasor of window k alone.
3. eps_k = ||w_k - w_{k-1}|| / ||w_k||.  For a geometrically decaying transient
   with ratio rho = eps_k / eps_{k-1}, the remaining error of w_k is
   ~ eps_k * rho / (1 - rho).  Stop when that estimate (and eps_k) is below
   --tol for three windows in a row (rho from a least-squares fit of log eps
   over the last 10 windows), or stop at --max-windows.
   The source is ramped on smoothly (--ramp, in periods): an abrupt start leaves
   a slowly drifting quasi-static field that the thin PML hardly absorbs, which
   made eps plateau near 1e-4 and decay only algebraically.  The steady state
   does not depend on how the source is switched on.
4. Only then the near-to-far box is created, one more window of Nt steps is
   accumulated, and the far field is projected.  The DFT over a whole number of
   periods of the periodic sequence is exact, so the far field carries only the
   remaining transient, below tol.

    python td_converged.py bw              # report Fig. 1 / Table 1 frequencies
    python td_converged.py bw --wh-tol 1e-10   # also the tight WaveHoltz reference
"""

import argparse
import json
import math
import time

import numpy as np
import meep as mp

import cw_common as W
from cw_common import C
import horn as H
import horn_cw as HC

OUT = W.RESULTS / "td_converged"


def simulate(a, f_ghz, dphi_e=0.0):
    f_ghz, dphi_e = float(f_ghz), float(dphi_e)
    p = H.params(2, a)
    fcen = C.ghz(f_ghz)
    courant, nt = W.courant_for(fcen, p["resolution"], a.periods)
    meta = dict(antenna="horn", solver="td_conv", dim=2, f_ghz=f_ghz, dphi_e=dphi_e,
                dphi_h=0.0, dpml=a.dpml, n2f_gap=a.n2f_gap, dang=a.dang, courant=courant,
                periods=a.periods, tol=a.tol, max_windows=a.max_windows, ramp=a.ramp, **p)
    tag = f"horn2d_tdconv_N{p['n']}_f{f_ghz:g}_E{dphi_e:g}"
    if mp.am_master():
        print(f"== Horn 2D [td, transient < {a.tol:g}] f={f_ghz} GHz dPhi={dphi_e} "
              f"Courant={courant:.4f} ({nt} steps / window of {a.periods} periods)")

    def compute():
        x0, cell = H.geometry_setup(a, 2, p)
        geom, rads = H.layout(p, 2, x0)
        src = [mp.Source(mp.ContinuousSource(frequency=fcen, width=a.ramp / fcen),
                         component=mp.Ey, center=c,
                         size=s, amplitude=C.element_phase(i, j, dphi_e, 0.0))
               for i, j, c, s in rads]
        sim = mp.Simulation(cell_size=cell, resolution=p["resolution"], geometry=geom,
                            sources=src, boundary_layers=[mp.PML(a.dpml)], Courant=courant)
        inner = mp.Vector3(cell.x - 2 * a.dpml, cell.y - 2 * a.dpml)
        mon = sim.add_dft_fields([mp.Ey, mp.Hz], fcen, 0, 1, center=mp.Vector3(), size=inner,
                                 yee_grid=True, decimation_factor=1)
        sim.init_sim()
        sim._evaluate_dft_objects()
        t0 = time.time()

        def total():
            return np.concatenate([np.ravel(sim.get_dft_array(mon, c, 0)) for c in (mp.Ey, mp.Hz)])

        prev_total = np.zeros_like(total())
        w_prev, eps_hist, est, ok_count, k = None, [], np.inf, 0, 0
        for k in range(1, a.max_windows + 1):
            for _ in range(nt):
                sim.fields.step()
            tot = total()
            w = tot - prev_total
            prev_total = tot
            if w_prev is not None:
                eps = np.linalg.norm(w - w_prev) / np.linalg.norm(w)
                eps_hist.append(eps)
                if len(eps_hist) >= 10:
                    # decay ratio per window from a least-squares fit of log(eps) over 10 windows
                    rho = float(np.exp(np.polyfit(np.arange(10), np.log(eps_hist[-10:]), 1)[0]))
                    est = eps * rho / (1 - rho) if rho < 1 else np.inf
                    ok_count = ok_count + 1 if (est < a.tol and eps < a.tol) else 0
                    if mp.am_master() and (k % 20 == 0 or ok_count):
                        print(f"   window {k:5d} ({k * a.periods} periods): eps {eps:.2e} "
                              f"rho {rho:.4f} est {est:.2e}")
                    if ok_count >= 3:
                        break
            w_prev = w
        conv_s = time.time() - t0
        converged = ok_count >= 3
        n2f = sim.add_near2far(fcen, 0, 1, *C.near2far_box(sim.cell_size, a.dpml, a.n2f_gap))
        sim._evaluate_dft_objects()
        for _ in range(nt):
            sim.fields.step()
        out = C.sim_farfield(sim, n2f, "E", a.dang)
        wall = time.time() - t0
        stats = dict(converged=converged, windows=k + 1, periods=(k + 1) * a.periods,
                     timesteps=(k + 1) * nt, eps_last=float(eps_hist[-1]) if eps_hist else None,
                     est_error=float(est), transient_s=conv_s, wall_s=wall,
                     eps_history=[float(e) for e in eps_hist])
        if mp.am_master():
            print(f"   td: converged={converged} after {(k + 1) * a.periods} periods "
                  f"({(k + 1) * nt} steps), est. transient {est:.1e}, {wall:.1f} s")
        out["stats_json"] = json.dumps(stats)
        out["meta"] = meta
        sim.reset_meep()
        return out

    return C.cached(C.cache_path(a.outdir, tag, meta), compute, a.force)


def wh_tight(f, tol):
    """WaveHoltz reference at a tighter GMRES tolerance (cached under results/runs)."""
    a, _ = HC.parse_args(["run", "--solvers", "wh", "--show", "", "--tol", f"{tol:g}",
                          "--maxiters-wh", "5000"])
    return HC.simulate(a, "wh", f, 0.0)


def rel_far(r, ref):
    """max |P - P_ref| / max P_ref over all angles (same normalisation: same window)."""
    return float(np.max(np.abs(r["P_E"] - ref["P_E"])) / np.max(ref["P_E"]))


def cmd_bw(a):
    rows = []
    for f in HC.BW_FREQS:
        r = simulate(a, f)
        st = W.stats_of(r)
        wh6 = wh_tight(f, 1e-6)
        whT = wh_tight(f, a.wh_tol) if a.wh_tol else None
        if mp.am_master():
            row = dict(f=f, periods=st["periods"], steps=st["timesteps"], wall_s=st["wall_s"],
                       est=st["est_error"], converged=st["converged"],
                       bw=HC.metric(r)["bw_deg"], bw_wh6=HC.metric(wh6)["bw_deg"],
                       dP_wh6=HC.pattern_diff_db(r, wh6), rel_wh6=rel_far(r, wh6))
            if whT is not None:
                sT = W.stats_of(whT)
                row.update(bw_whT=HC.metric(whT)["bw_deg"], dP_whT=HC.pattern_diff_db(r, whT),
                           rel_whT=rel_far(r, whT), whT_iters=sT["iters"],
                           whT_wall=sT["wall_s"], rel_wh6_whT=rel_far(wh6, whT))
            rows.append(row)
            print("   " + ", ".join(f"{k} {v:.3e}" if isinstance(v, float) else f"{k} {v}"
                                    for k, v in row.items()))
    if mp.am_master():
        (OUT / "td_converged_bw.json").write_text(json.dumps(rows, indent=1))
        print("wrote", OUT / "td_converged_bw.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["bw", "run"])
    ap.add_argument("--freq", type=float, default=3.0)
    ap.add_argument("--tol", type=float, default=1e-6, help="transient tolerance")
    ap.add_argument("--periods", type=int, default=2, help="window length (as WaveHoltz)")
    ap.add_argument("--max-windows", type=int, default=1000,
                    help="cap on the run length, in windows (default 1000 = 2000 periods)")
    ap.add_argument("--ramp", type=float, default=3.0,
                    help="ContinuousSource turn-on width in periods (0 = abrupt, as horn.py)")
    ap.add_argument("--wh-tol", type=float, default=None,
                    help="also compare with WaveHoltz at this GMRES tolerance")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    # parent-geometry options (horn.py defaults)
    a.n = a.wall = a.pad = a.resolution = a.sx = a.sy = a.x0 = None
    a.no_backplane, a.dpml, a.n2f_gap, a.dang, a.outdir = False, 1.0, 0.5, 0.25, OUT
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "bw":
        cmd_bw(a)
    else:
        simulate(a, a.freq)


if __name__ == "__main__":
    main()
