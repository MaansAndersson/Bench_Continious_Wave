"""Has the time-domain solver converged?  TD run length vs the WaveHoltz solution.

The previous report ran horn.py for max(6.67 ns, 10 periods), with the
source switched on abruptly and the near-to-far DFT accumulated from t = 0.
Here the same TD computation (horn.py, imported unchanged) is repeated for
10, 20, 40, 80 and 160 periods and compared with the cached WaveHoltz solution
of the same configuration (solve_wh, tol 1e-6; it agrees with
solve_cw to ~1e-5 dB, so it serves as the converged reference).

Metrics per run length: max pattern difference (dB, above -30 dB), beamwidth,
beam angle, and the scan loss P_peak(80) - P_peak(0) where both are available.

    python td_convergence.py           # runs what is missing, then plots
"""

import argparse
import sys

import numpy as np
import meep as mp

import cw_common as W
from cw_common import C
import horn as H
import horn_cw as HC

PERIODS = (10, 20, 40, 80, 160)
CASES = [(0.5, 0.0), (0.5, 80.0), (1.0, 0.0), (1.0, 80.0), (3.0, 0.0)]
OUT = W.RESULTS / "td_convergence"


def td_args(periods, force=False):
    """Namespace for horn.py with run length = `periods` periods exactly."""
    return argparse.Namespace(
        resolution=None, dpml=1.0, n2f_gap=0.5, until_ns=0.0, min_periods=periods,
        dang=0.25, complex=False, no_subpixel=False, force=force, outdir=OUT, n=None,
        wall=None, pad=None, sx=None, sy=None, x0=None, no_backplane=False,
        source_width=0.0)


def wh_ref(f, d):
    a, _ = HC.parse_args(["run", "--solvers", "", "--show", "wh"])
    return HC.simulate(a, "wh", f, d, 0.0, 2, only_cached=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    opt = ap.parse_args()
    td = {(f, d, n): H.simulate(td_args(n, opt.force), f, d) for f, d in CASES for n in PERIODS}
    ref = {(f, d): wh_ref(f, d) for f, d in CASES}
    if not mp.am_master():
        return
    missing = [c for c, r in ref.items() if r is None]
    if missing:
        print("missing WaveHoltz reference for", missing, "- run horn_cw.py first")

    rows = []
    for f, d in CASES:
        r = ref[f, d]
        mr = HC.metric(r) if r is not None else None
        for n in PERIODS:
            t = td[f, d, n]
            mt = HC.metric(t)
            dp = HC.pattern_diff_db(t, r) if r is not None else np.nan
            rows.append((f, d, n, dp, mt["bw_deg"], mt["peak_deg"],
                         mr["bw_deg"] if mr else np.nan, mr["peak_deg"] if mr else np.nan))
    lines = ["f_GHz dPhi periods  dP_vs_wh_dB  BW_td    BW_wh   beam_td  beam_wh"]
    lines += [f"{f:5.2f} {d:4.0f} {n:7d}  {dp:10.3f}  {bw:7.3f}  {bwr:7.3f}  {pk:7.2f}  {pkr:7.2f}"
              for f, d, n, dp, bw, pk, bwr, pkr in rows]

    # scan loss vs run length
    sl_lines = ["f_GHz periods  SL_td   (SL_wh)"]
    sl = {}
    for f in (0.5, 1.0):
        r0, r80 = ref[f, 0.0], ref[f, 80.0]
        slw = (10 * np.log10(HC.metric(r80)["peak_power"] / HC.metric(r0)["peak_power"])
               if r0 is not None and r80 is not None else np.nan)
        for n in PERIODS:
            s = 10 * np.log10(HC.metric(td[f, 80.0, n])["peak_power"] /
                              HC.metric(td[f, 0.0, n])["peak_power"])
            sl[f, n] = (s, slw)
            sl_lines.append(f"{f:5.2f} {n:7d}  {s:6.2f}  ({slw:6.2f})")
    txt = "\n".join(lines + [""] + sl_lines) + "\n"
    print(txt)
    (OUT / "td_convergence.txt").write_text(txt)

    plt = C.pyplot()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for f, d in CASES:
        dps = [x[3] for x in rows if x[0] == f and x[1] == d]
        axs[0].loglog(PERIODS, dps, "o-", mfc="none", label=f"{f} GHz, $\\Delta\\Phi$ = {d:g}")
        bwerr = [abs(x[4] - x[6]) for x in rows if x[0] == f and x[1] == d]
        axs[1].loglog(PERIODS, np.maximum(bwerr, 1e-4), "o-", mfc="none",
                      label=f"{f} GHz, $\\Delta\\Phi$ = {d:g}")
    pp = np.array(PERIODS, float)
    for ax in axs[:2]:
        ax.loglog(pp, pp[0] / pp * ax.get_ylim()[1] * 0.5, "k:", lw=0.8, label="$\\propto 1/T$")
        ax.set_xlabel("TD run length (periods)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    axs[0].set_ylabel("max |$\\Delta$P| vs WaveHoltz, above -30 dB (dB)")
    axs[0].set_title("Pattern error of the TD solution", fontsize=9)
    axs[1].set_ylabel("|BW$_{td}$ - BW$_{wh}$| (deg)")
    axs[1].set_title("Beamwidth error", fontsize=9)
    for f, mk in ((0.5, "o"), (1.0, "s")):
        axs[2].semilogx(PERIODS, [sl[f, n][0] for n in PERIODS], mk + "-", mfc="none",
                        label=f"TD, {f} GHz")
        axs[2].axhline(sl[f, PERIODS[0]][1], ls="--", lw=0.8,
                       color=axs[2].lines[-1].get_color(), label=f"WaveHoltz, {f} GHz")
    axs[2].set_xlabel("TD run length (periods)")
    axs[2].set_ylabel("scan loss at $\\Delta\\Phi$ = 80 (dB)")
    axs[2].set_title("Scan loss", fontsize=9)
    axs[2].grid(alpha=0.3, which="both")
    axs[2].legend(fontsize=7)
    C.save(fig, OUT / "td_convergence.png")


if __name__ == "__main__":
    main()
