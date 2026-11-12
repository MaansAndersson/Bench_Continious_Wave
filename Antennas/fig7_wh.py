"""Paper Fig. 7 at t = infinity: the WaveHoltz (solve_wh) steady state of the horn array.

horn.py fig7 shows Ey at t = 0.5, 1, 1.5, 2 ns after an abrupt switch-on (2.5 GHz,
dPhi = -35 deg, N = 16, 80 x 150 cm cell, array at x0 = -32 cm) and the far field after
6.67 ns.  Here the same simulation is solved for the time-periodic state with solve_wh, i.e.
the field as t -> infinity, at the paper's 6 pixels/cm and 0.2 cm wall.  The snapshot is the
real Ey right after the solve (t = 0 mod the 2-period window), in the same units as
horn.py fig7.  Its colour scale is the 99th percentile of |Ey| in front of the array
(~0.1): the paper's +-0.002, made for the arriving wavefront, saturates the steady state.  The
far field comes from the DFT over one window.  The cached time-domain far field of horn.py fig7 is overlaid.

    python fig7_wh.py      # results/cw/fig7_wh/horn_fig7_wh_{field,pattern}.png (+ cached run)
"""

import glob
import json
import sys

import numpy as np
import meep as mp

import cw_common as W
from cw_common import C
import horn_cw as HC

F, DPHI = 2.5, -35.0
OUT = W.RESULTS / "fig7_wh"


def args(extra=()):
    a, _ = HC.parse_args(["run", "--solvers", "wh", "--show", "", "--resolution", "6",
                          "--sx", "80", "--sy", "150", "--x0", "-32", "--outdir", str(OUT),
                          "--save-fields", *extra])
    return a


def td_fig7():
    """The horn.py fig7 run at the paper's settings (or None)."""
    for fn in sorted(glob.glob(str(W.TD_RUNS / "horn2d_fig7_*.npz"))):
        with np.load(fn, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
        out["meta"] = json.loads(str(out["meta"]))
        if out["meta"]["resolution"] == 6 and out["meta"]["sx"] == 80.0:
            return out
    return None


def main(argv):
    a = args(argv)
    r = HC.simulate(a, "wh", F, DPHI)
    if not mp.am_master():
        return
    st = W.stats_of(r)
    td = td_fig7()
    m_wh = C.report(r)[0]
    plt = C.pyplot()
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(4.6, 6.6))
    ext = [-a.sx / 2, a.sx / 2, -a.sy / 2, a.sy / 2]
    ey = r["Ey_t0"]
    vmax = float(np.percentile(np.abs(ey[r["x"] > a.x0 + 18, :]), 99))   # in front of the mouths
    im = ax.imshow(ey.T, origin="lower", extent=ext, cmap="cool", vmin=-vmax, vmax=vmax)
    metal = np.ma.masked_where(np.abs(r["eps"] - 1) < 1e-6, np.ones_like(r["eps"]))
    ax.imshow(metal.T, origin="lower", extent=ext, cmap=ListedColormap(["k"]), interpolation="nearest")
    ax.set_title(f"$t=\\infty$ (solve_wh, {st['iters']} GMRES iters)", fontsize=9)
    ax.set_xlabel("x (cm)", fontsize=8)
    ax.set_ylabel("y (cm)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.08, label="Ey")
    fig.savefig(OUT / "horn_fig7_wh_field.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT / "horn_fig7_wh_field.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 6.8), subplot_kw={"projection": "polar"})
    C.pattern_panel(ax, r, "E", f"N = 16 horns, {F} GHz, $\\Delta\\Phi$ = {DPHI:g} deg, 6 px/cm")
    ax.lines[-1].set_label(f"solve_wh ($t=\\infty$): beam {m_wh['peak_deg']:.2f}, "
                           f"BW {m_wh['bw_deg']:.2f} deg")
    if td is not None:
        m_td = C.beam_metrics(td["angles"], td["P_E"])
        p = np.maximum(td["P_E"], 1e-300)
        db = np.clip(10 * np.log10(p / m_td["peak_power"]), -40, 0)
        ang = np.deg2rad(np.append(td["angles"], td["angles"][0]))
        ax.plot(ang, np.append(db, db[0]), "k--", lw=0.8,
                label=f"time domain, {td['meta']['until'] / C.ns(1):.2f} ns: beam "
                      f"{m_td['peak_deg']:.2f}, BW {m_td['bw_deg']:.2f} deg")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.savefig(OUT / "horn_fig7_wh_pattern.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT / "horn_fig7_wh_pattern.png")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
