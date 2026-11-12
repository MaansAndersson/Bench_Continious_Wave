"""Horn radiator: line current across the box (default) vs point dipole (--radiator point).

Fig. 3 of the paper draws the horn radiator as a dot; the text of Sec. 4 speaks of a
"proper sized radiator".  This runs every 2D horn configuration of the reports with a
point dipole, with both the short time-domain runs of horn.py (6.67 ns / 10 periods,
as in the first report) and WaveHoltz, and compares them with the cached line-source
results.

The point-source runs are cached under results/point/ (WaveHoltz) and
results/td/point/ (time domain), apart from the line-source caches, so that no lookup can
mix them up.

    python radiator_compare.py          # runs what is missing, then figures + tables
"""

import json
import time

import numpy as np
import meep as mp

import cw_common as W
from cw_common import C
import horn as H
import horn_cw as HC
import timing as T

POINT_TD = W.TD_RESULTS / "point"
POINT_WH = W.RESULTS / "point"
FIG6 = [(f, d) for f in (0.5, 5.0) for d in (0.0, 10.0, 20.0, 30.0)]


def horn_configs():
    return [(f, d) for ant, n, f, d in T.configs() if ant == "horn"]


def td(f, d, point):
    if not point:
        return W.td_result(antenna="horn", dim=2, n=16, f_ghz=f, dphi_e=d, dphi_h=0.0,
                           resolution=6, dpml=1.0), None
    a = T.td_ns("horn", POINT_TD)
    a.force, a.radiator = False, "point"
    t0 = time.time()
    r = H.simulate(a, f, d)
    return r, time.time() - t0


def wh(f, d, point):
    if not point:
        a, _ = HC.parse_args(["run", "--solvers", "", "--show", "wh"])
        return HC.simulate(a, "wh", f, d, only_cached=True)
    a, _ = HC.parse_args(["run", "--solvers", "wh", "--show", "", "--outdir", str(POINT_WH)])
    a.radiator = "point"
    return HC.simulate(a, "wh", f, d)


def run_all():
    out = {}
    for f, d in horn_configs():
        rt, wall = td(f, d, True)
        rw = wh(f, d, True)
        out[f, d] = dict(td_line=td(f, d, False)[0], td_point=rt, td_point_s=wall,
                         wh_line=wh(f, d, False), wh_point=rw)
    return out


def m(r):
    return HC.metric(r)


def polar(ax, res, f, d, pairs, title, rmin=-40.0):
    first = res["wh_line"]
    meta = first["meta"]
    th = C.array_factor(first["angles"], meta["n"], C.d_over_lambda(meta), d)
    ang = np.deg2rad(np.append(first["angles"], first["angles"][0]))

    def db(p, ref):
        x = np.clip(10 * np.log10(np.maximum(p, 1e-300) / ref), rmin, 0)
        return np.append(x, x[0])
    ax.plot(ang, db(th, 1.0), color="r", lw=0.7, label="Eq. (25)")
    for key, style, lab in pairs:
        r = res[key]
        ax.plot(ang, db(r["P_E"], m(r)["peak_power"]), label=lab, **style)
    ax.set_ylim(rmin, 0)
    ax.set_yticks([-30, -15, -3, 0])
    ax.set_yticklabels(["-30", "-15", "-3", "0"], fontsize=7)
    ax.set_thetagrids(range(0, 360, 45), fontsize=7)
    ax.set_title(title, fontsize=8)


def figures(res):
    plt = C.pyplot()
    sets = {
        "wh": [("wh_line", dict(color="tab:blue", lw=1.3), "WaveHoltz, line"),
               ("wh_point", dict(color="tab:orange", lw=1.1, ls="--"), "WaveHoltz, dot")],
        "td": [("td_line", dict(color="0.35", lw=1.3), "TD (10 periods), line"),
               ("td_point", dict(color="k", lw=1.0, ls=":"), "TD (10 periods), dot")],
    }
    for name, pairs in sets.items():
        fig, axs = plt.subplots(4, 2, subplot_kw={"projection": "polar"}, figsize=(7.2, 14))
        for ax, (f, d) in zip(axs.flat, FIG6):
            polar(ax, res[f, d], f, d, pairs, f"{f:g} GHz, $\\Delta\\Phi$ = {d:g} deg")
        axs[0, 0].legend(fontsize=6, loc="lower left")
        fig.suptitle("Fig. 6 of the paper, horn radiator: line vs dot "
                     f"({'WaveHoltz' if name == 'wh' else 'time domain, 10 periods'})",
                     fontsize=9)
        fig.tight_layout()
        fn = W.RESULTS / f"radiator_fig6_{name}.png"
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        print("wrote", fn)


def tables(res):
    """LaTeX rows: Fig. 6 cases, and scan loss, for line vs dot."""
    gen = W.REPORTS / "cw" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, d in FIG6:
        r = res[f, d]
        rows.append(
            f"{f:.1f} & {d:.0f} & {m(r['wh_line'])['peak_deg']:.2f} & {m(r['wh_point'])['peak_deg']:.2f} & "
            f"{m(r['wh_line'])['bw_deg']:.2f} & {m(r['wh_point'])['bw_deg']:.2f} & "
            f"{HC.pattern_diff_db(r['wh_point'], r['wh_line']):.2f} & "
            f"{HC.pattern_diff_db(r['td_point'], r['td_line']):.2f} & "
            f"{HC.pattern_diff_db(r['td_point'], r['wh_point']):.2f} \\\\")
    (gen / "tab_radiator_fig6.tex").write_text("\n".join(rows) + "\n")

    def sl(r0, r80):
        return 10 * np.log10(m(r80)["peak_power"] / m(r0)["peak_power"])
    paper = {0.5: -11.6, 1.0: -1.2, 2.0: -1.0, 4.0: -0.9}
    rows = []
    for f in (0.5, 1.0, 2.0, 4.0):
        a, b = res[f, 0.0], res[f, 80.0]
        rows.append(f"{f:.1f} & {sl(a['td_line'], b['td_line']):.2f} & {sl(a['td_point'], b['td_point']):.2f} & "
                    f"{sl(a['wh_line'], b['wh_line']):.2f} & {sl(a['wh_point'], b['wh_point']):.2f} & "
                    f"{paper[f]:.1f} \\\\")
    (gen / "tab_radiator_scan.tex").write_text("\n".join(rows) + "\n")

    # summary over all horn configurations
    summ = {}
    for key, ref in (("wh_point", "wh_line"), ("td_point", "td_line")):
        dbw = [abs(m(r[key])["bw_deg"] - m(r[ref])["bw_deg"]) for (f, d), r in res.items() if d <= 30]
        dp = [HC.pattern_diff_db(r[key], r[ref]) for r in res.values()]
        summ[key] = dict(max_dbw=max(dbw), median_dp=float(np.median(dp)), max_dp=max(dp))
    (W.RESULTS / "radiator_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def main():
    res = run_all()
    if mp.am_master():
        figures(res)
        tables(res)


if __name__ == "__main__":
    main()
