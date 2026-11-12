"""Sensitivity to the metal thickness, which the paper does not specify.

Yagi-Uda wire width (elements and boom) and horn wall thickness are free choices
(0.2 cm by default, about one pixel at the paper's resolution of 6/cm).  This reruns the
cases where our results differ from the paper with 0.2, 0.3 and 0.4 cm, using WaveHoltz
(converged solutions, so no time-stepping transient enters the comparison):

  Yagi  N = 8 and 16, 5 GHz, dPhi = 0..80 deg      (Fig. 4 top: steering slope, BW)
  horn  N = 16, dPhi = 0, 0.3..3 GHz                (Fig. 4 bottom right: BW = a/f + b)
  horn  N = 16, dPhi = 0 and 80 deg, 0.5..4 GHz     (Table 1: scan loss)

0.2 cm is the default and comes from the main cache (results/runs); the other thicknesses
are cached under results/thickness/.  The horn radiator spans the box interior, so its
length is a - wall.

    python thickness_sensitivity.py
"""

import json

import numpy as np
import meep as mp

import cw_common as W
from cw_common import C
import horn_cw as HC
import yagi_cw as YC
import yagi_uda as Y

# 0.1 cm is not included: below one pixel (1/6 cm) a PEC sheet is transparent at some
# sub-pixel offsets at resolution 6 (wall_transparency.py), so it would test the grid, not
# the thickness.
THICK = (0.2, 0.3, 0.4)
DEFAULT = 0.2
OUT = W.RESULTS / "thickness"
GEN = W.REPORTS / "cw" / "generated"
DPS = np.arange(0, 81, 10.0)
SCAN_F = (0.5, 1.0, 2.0, 4.0)
PAPER_SL = {0.5: -11.6, 1.0: -1.2, 2.0: -1.0, 4.0: -0.9}
STYLE = {0.2: dict(color="tab:blue", marker="o"), 0.3: dict(color="tab:green", marker="v"),
         0.4: dict(color="tab:red", marker="^")}


def outdir(t):
    return W.RESULTS if t == DEFAULT else OUT


def yagi(t, n, d):
    a, _ = YC.parse_args(["run", "--solvers", "wh", "--show", "", "--wire", f"{t:g}",
                          "--outdir", str(outdir(t))])
    return YC.simulate(a, "wh", n, 5.0, d)


def horn(t, f, d):
    a, _ = HC.parse_args(["run", "--solvers", "wh", "--show", "", "--outdir", str(outdir(t))])
    if t != DEFAULT:
        a.wall = t
    return HC.simulate(a, "wh", f, d)


def run_all():
    res = {}
    for t in THICK:
        for n in (8, 16):
            for d in DPS:
                res["yagi", t, n, d] = yagi(t, n, d)
        for f in HC.BW_FREQS:
            res["horn", t, f, 0.0] = horn(t, f, 0.0)
        for f in SCAN_F:
            for d in (0.0, 80.0):
                res["horn", t, f, d] = horn(t, f, d)
    return res


def m(r):
    return HC.metric(r)


def scan_loss(res, t, f):
    return 10 * np.log10(m(res["horn", t, f, 80.0])["peak_power"] /
                         m(res["horn", t, f, 0.0])["peak_power"])


def analyse(res):
    slope_th = C.linear_slope(Y.d_over_lambda(5.0))
    out = {"yagi": {}, "bw": {}, "scan": {}, "dP": {}}
    for t in THICK:
        for n in (8, 16):
            ang = np.array([abs(m(res["yagi", t, n, d])["peak_deg"]) for d in DPS])
            out["yagi"][f"{t}_{n}"] = dict(slope=float(np.polyfit(DPS, ang, 1)[0]),
                                           bw0=m(res["yagi", t, n, 0.0])["bw_deg"],
                                           theory=slope_th)
        bw = [m(res["horn", t, f, 0.0])["bw_deg"] for f in HC.BW_FREQS]
        (ka, kb), (sa, sb) = C.fit_inverse_f(HC.BW_FREQS, bw)
        out["bw"][str(t)] = dict(a=ka, sa=sa, b=kb, sb=sb, bw=bw)
        out["scan"][str(t)] = {str(f): scan_loss(res, t, f) for f in SCAN_F}
        if t != DEFAULT:
            yd = [HC.pattern_diff_db(res["yagi", t, n, d], res["yagi", DEFAULT, n, d])
                  for n in (8, 16) for d in DPS]
            hd = [HC.pattern_diff_db(res["horn", t, f, 0.0], res["horn", DEFAULT, f, 0.0])
                  for f in HC.BW_FREQS]
            dbw = [abs(x - y) for x, y in zip(bw, out["bw"][str(DEFAULT)]["bw"])] \
                if str(DEFAULT) in out["bw"] else []
            out["dP"][str(t)] = dict(yagi_median=float(np.median(yd)), yagi_max=float(max(yd)),
                                     horn_median=float(np.median(hd)), horn_max=float(max(hd)),
                                     horn_max_dbw=float(max(dbw)) if dbw else None)
    # beamwidth change needs the default first; recompute now that all are known
    for t in THICK:
        if t != DEFAULT:
            out["dP"][str(t)]["horn_max_dbw"] = float(max(
                abs(x - y) for x, y in zip(out["bw"][str(t)]["bw"], out["bw"][str(DEFAULT)]["bw"])))
    return out


def figure(res, out):
    plt = C.pyplot()
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    slope_th = C.linear_slope(Y.d_over_lambda(5.0))
    xx = np.linspace(0, 85, 10)
    for t in THICK:
        for n, ls in ((8, "--"), (16, "-")):
            ang = [abs(m(res["yagi", t, n, d])["peak_deg"]) for d in DPS]
            st = STYLE[t]
            axs[0].plot(DPS, ang, ls=ls, marker=st["marker"], color=st["color"], mfc="none", ms=4,
                        lw=0.9, label=f"N = {n}, wire {t:g} cm: slope "
                                      f"{out['yagi'][f'{t}_{n}']['slope']:.4f}")
    axs[0].plot(xx, slope_th * xx, color="k", lw=0.8, ls=":", label=f"theory {slope_th:.4f}")
    axs[0].set_xlabel(r"$\Delta\Phi$ (deg)")
    axs[0].set_ylabel(r"$|\Delta\phi|$ (deg)")
    axs[0].set_title("Yagi-Uda steering at 5 GHz vs wire width", fontsize=9)
    axs[0].legend(fontsize=6)
    for t in THICK:
        b = out["bw"][str(t)]
        axs[1].plot(HC.BW_FREQS, b["bw"], marker=STYLE[t]["marker"], color=STYLE[t]["color"],
                    mfc="none", lw=0.9, label=f"wall {t:g} cm: a = {b['a']:.2f}, b = {b['b']:.2f}")
    axs[1].set_xlabel("Frequency (GHz)")
    axs[1].set_ylabel("Beam width (deg)")
    axs[1].set_title("Horn N = 16 beamwidth vs wall thickness (fit a/f + b)", fontsize=9)
    axs[1].legend(fontsize=7)
    for t in THICK:
        axs[2].plot(SCAN_F, [out["scan"][str(t)][str(f)] for f in SCAN_F],
                    marker=STYLE[t]["marker"], color=STYLE[t]["color"], mfc="none", lw=0.9,
                    label=f"wall {t:g} cm")
    axs[2].plot(SCAN_F, [PAPER_SL[f] for f in SCAN_F], "ks", label="paper")
    axs[2].set_xscale("log")
    axs[2].set_xticks(SCAN_F)
    axs[2].set_xticklabels([f"{f:g}" for f in SCAN_F])
    axs[2].xaxis.set_minor_formatter(plt.NullFormatter())
    axs[2].set_xlabel("Frequency (GHz)")
    axs[2].set_ylabel(r"scan loss at $\Delta\Phi$ = 80 (dB)")
    axs[2].set_title("Horn scan loss vs wall thickness", fontsize=9)
    axs[2].legend(fontsize=7)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle("Sensitivity to the unspecified metal thickness (WaveHoltz, converged)",
                 fontsize=10)
    fig.tight_layout()
    fn = W.RESULTS / "thickness_sensitivity.png"
    fig.savefig(fn, dpi=150)
    print("wrote", fn)


def tables(out):
    GEN.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in THICK:
        y8, y16 = out["yagi"][f"{t}_8"], out["yagi"][f"{t}_16"]
        b = out["bw"][str(t)]
        dp = out["dP"].get(str(t))
        rows.append(f"{t:g} & {y8['slope']:.4f} & {y8['bw0']:.2f} & {y16['slope']:.4f} & "
                    f"{y16['bw0']:.2f} & ${b['a']:.2f}\\pm{b['sa']:.2f}$ & ${b['b']:.2f}\\pm{b['sb']:.2f}$ & "
                    + (f"{dp['yagi_median']:.2f} / {dp['yagi_max']:.2f} & "
                       f"{dp['horn_median']:.2f} / {dp['horn_max']:.2f}" if dp else "ref. & ref.")
                    + " \\\\")
    (GEN / "tab_thick.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for f in SCAN_F:
        rows.append(f"{f:.1f} & " + " & ".join(f"{out['scan'][str(t)][str(f)]:.2f}" for t in THICK)
                    + f" & {PAPER_SL[f]:.1f} \\\\")
    (GEN / "tab_thick_scan.tex").write_text("\n".join(rows) + "\n")
    (W.RESULTS / "thickness_sensitivity.json").write_text(json.dumps(out, indent=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = run_all()
    if mp.am_master():
        out = analyse(res)
        figure(res, out)
        tables(out)
        print(json.dumps({k: v for k, v in out.items() if k != "bw"}, indent=1))


if __name__ == "__main__":
    main()
