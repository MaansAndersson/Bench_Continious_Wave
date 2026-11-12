"""Rebuild figures, LaTeX tables and report/cw_report.pdf from cached runs only.

Nothing is simulated: every figure command runs with --solvers '' (show cached
cw and wh results, compute none).  Run it again whenever more solve_cw (or
WaveHoltz) results have been cached; the report says how many are in.

    python make_report.py            # figures + tables + pdflatex
    python make_report.py --no-pdf   # figures + tables only
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime

import numpy as np

import cw_common as W
from cw_common import C
import horn_cw as HC
import yagi_cw as YC
import horn as H

REPORT = W.REPORTS / "cw"
TEXLABEL = {"td": "time domain", "cw": r"\texttt{solve\_cw} (BiCGSTAB-10)",
            "wh": r"WaveHoltz \texttt{solve\_wh} (GMRES(50), 2 periods)"}
GEN = REPORT / "generated"


def timing():
    """results/timing.json (timing.py): idle-machine wall times; {} if absent."""
    import json
    src = W.RESULTS / "timing.json"
    return json.loads(src.read_text()) if src.exists() else {}


TIM = None


def tkey(ant, n, f, d, periods=None):
    k = f"{ant}_N{n}_f{f:g}_E{d:g}"
    return k + (f"_P{periods}" if periods else "")


def tget(kind, k, field):
    v = TIM.get(kind, {}).get(k, {}).get(field)
    return v


def args_for(mod, cmd):
    a, fn = mod.parse_args([cmd, "--solvers", ""])
    return a, fn


def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def sci(x):
    if x is None or not np.isfinite(x):
        return "--"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def write(name, text):
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / name).write_text(text)


class Count:
    def __init__(self):
        self.n = {"wh": [0, 0], "cw": [0, 0], "td": [0, 0]}
        self.cost = {"wh": [], "cw": []}

    def add(self, res):
        for s in self.n:
            self.n[s][1] += 1
            if s in res:
                self.n[s][0] += 1
                if s != "td":
                    self.cost[s].append(W.stats_of(res[s]))


def dP(res, s, ref, plane="E"):
    return HC.pattern_diff_db(res[s], res[ref], plane) if s in res and ref in res else None


def bw_of(res, s, plane="E"):
    return HC.metric(res[s], plane)["bw_deg"] if s in res else None


def peak_of(res, s, plane="E"):
    return HC.metric(res[s], plane)["peak_deg"] if s in res else None


# --------------------------------------------------------------------------

def horn_bw(cnt):
    a, _ = args_for(HC, "bw")
    p = H.params(2, a)
    rows, data = [], {}
    for f in HC.BW_FREQS:
        r = HC.all_results(a, f, 0.0)
        cnt.add(r)
        data[f] = r
        th = C.theory_metrics(p["n"], H.dl(p, f), 0)["bw_deg"]
        st = {s: W.stats_of(r[s]) if s in r else {} for s in ("wh", "cw")}
        k = tkey("horn", 16, f, 0.0)
        wh_s = tget("wh", k, "wall_s")
        rows.append(
            f"{f:.2f} & {th:.2f} & {fmt(bw_of(r, 'td'), 2)} & {fmt(bw_of(r, 'wh'), 3)} & "
            f"{fmt(bw_of(r, 'cw'), 3)} & {fmt(tget('td', k, 'periods'), 0)} & "
            f"{fmt(tget('td', k, 'wall_s'), 1)} & {st['wh'].get('iters', '--')} & "
            f"{fmt(wh_s, 1)} & {st['cw'].get('iters', '--')} & "
            f"{fmt(st['cw'].get('wall_s'), 0)} & {fmt(dP(r, 'wh', 'td'), 2)} & "
            f"{sci(dP(r, 'wh', 'cw'))} \\\\")
    write("tab_bw.tex", "\n".join(rows) + "\n")
    fits = []
    for s in ("td", "wh", "cw"):
        fs = [f for f in HC.BW_FREQS if s in data[f]]
        if len(fs) >= 3:
            (ka, kb), (sa, sb) = C.fit_inverse_f(fs, [bw_of(data[f], s) for f in fs])
            fits.append(f"{TEXLABEL[s]} & {len(fs)} & ${ka:.2f}\\pm{sa:.2f}$ & "
                        f"${kb:.2f}\\pm{sb:.2f}$ \\\\")
        else:
            fits.append(f"{TEXLABEL[s]} & {len(fs)} & -- & -- \\\\")
    fits.append("paper (Hanson 2021) & 12 & $12.0\\pm0.1$ & $1.1\\pm0.2$ \\\\")
    write("tab_bw_fit.tex", "\n".join(fits) + "\n")


def horn_steer(cnt):
    a, _ = args_for(HC, "fig4")
    p = H.params(2, a)
    rows = []
    for f in (0.3, 1.5, 3.0):
        for d in (10.0, 20.0):
            r = HC.all_results(a, f, d)
            cnt.add(r)
            rows.append(f"{f:.1f} & {d:.0f} & {C.steer_angle(d, H.dl(p, f)):.3f} & "
                        f"{fmt(peak_of(r, 'td'))} & {fmt(peak_of(r, 'wh'))} & "
                        f"{fmt(peak_of(r, 'cw'))} \\\\")
        cnt.add(HC.all_results(a, f, 0.0))
    write("tab_horn_steer.tex", "\n".join(rows) + "\n")


def horn_scan(cnt):
    a, _ = args_for(HC, "table1")
    paper = {0.5: -11.6, 1.0: -1.2, 2.0: -1.0, 4.0: -0.9}
    rows = []
    for f in (0.5, 1.0, 2.0, 4.0):
        r0, r80 = HC.all_results(a, f, 0.0), HC.all_results(a, f, 80.0)
        cnt.add(r0)
        cnt.add(r80)

        def sl(s):
            if s in r0 and s in r80:
                return 10 * np.log10(HC.metric(r80[s])["peak_power"] / HC.metric(r0[s])["peak_power"])
            return None
        rows.append(f"{f:.1f} & {fmt(sl('td'), 2)} & {fmt(sl('wh'), 2)} & {fmt(sl('cw'), 2)} & "
                    f"{paper[f]:.1f} & {fmt(peak_of(r80, 'wh'), 2)} \\\\")
    write("tab_scan.tex", "\n".join(rows) + "\n")


def patterns(cnt, mod, cmd, cases, name):
    a, _ = args_for(mod, cmd)
    rows = []
    for case in cases:
        r = mod.all_results(a, *case)
        cnt.add(r)
        f, de = case[-2], case[-1]
        rows.append(f"{f:.1f} & {de:.0f} & {fmt(peak_of(r, 'td'), 2)} & {fmt(peak_of(r, 'wh'), 2)} & "
                    f"{fmt(peak_of(r, 'cw'), 2)} & {fmt(bw_of(r, 'td'), 2)} & {fmt(bw_of(r, 'wh'), 2)} & "
                    f"{fmt(bw_of(r, 'cw'), 2)} & {fmt(dP(r, 'wh', 'td'), 2)} & {sci(dP(r, 'wh', 'cw'))} \\\\")
    write(name, "\n".join(rows) + "\n")


def yagi_steer(cnt):
    a, _ = args_for(YC, "fig4")
    dps = np.arange(0, 81, 10.0)
    slope_th = C.linear_slope(YC.Y.TABLE1["dy"] / C.wavelength_cm(5.0))
    rows = []
    for n in (8, 16):
        res = {d: YC.all_results(a, n, 5.0, d) for d in dps}
        for d in dps:
            cnt.add(res[d])
        cells = []
        for s in ("td", "wh", "cw"):
            ds = [d for d in dps if s in res[d]]
            if len(ds) >= 2:
                ang = [abs(peak_of(res[d], s)) for d in ds]
                cells.append(f"{np.polyfit(ds, ang, 1)[0]:.4f}")
                cells.append(fmt(bw_of(res[0.0], s), 3) if s in res[0.0] else "--")
            else:
                cells += ["--", "--"]
        mx = [dP(res[d], "wh", "td") for d in dps if "wh" in res[d] and "td" in res[d]]
        mc = [dP(res[d], "wh", "cw") for d in dps if "wh" in res[d] and "cw" in res[d]]
        rows.append(f"{n} & {slope_th:.4f} & " + " & ".join(cells) +
                    f" & {fmt(max(mx) if mx else None, 2)} & {sci(max(mc)) if mc else '--'} \\\\")
    write("tab_yagi_steer.tex", "\n".join(rows) + "\n")


def td_convergence_tables():
    """Tables from results/td_convergence/td_convergence.txt (written by td_convergence.py)."""
    src = W.RESULTS / "td_convergence" / "td_convergence.txt"
    if not src.exists():
        write("tab_tdconv.tex", "\\multicolumn{10}{c}{not computed yet} \\\\\n")
        write("tab_tdconv_sl.tex", "\\multicolumn{4}{c}{not computed yet} \\\\\n")
        return
    main, sl = src.read_text().split("\n\n", 1)
    rows, last = [], None
    for ln in main.splitlines()[1:]:
        f, d, n, dp, bw, bwr, pk, pkr = ln.split()
        if last is not None and (f, d) != last:
            rows.append("\\midrule")
        last = (f, d)
        k = tkey("horn", 16, float(f), float(d))
        rows.append(f"{float(f):.1f} & {float(d):.0f} & {n} & "
                    f"{fmt(tget('td_conv', tkey('horn', 16, float(f), float(d), int(n)), 'wall_s'), 1)} & "
                    f"{fmt(tget('wh', k, 'wall_s'), 1)} & {float(dp):.3f} & {float(bw):.3f} & "
                    f"{float(bwr):.3f} & {float(pk):.2f} & {float(pkr):.2f} \\\\")
    write("tab_tdconv.tex", "\n".join(rows) + "\n")
    srows = []
    for ln in sl.strip().splitlines()[1:]:
        f, n, s_td, s_wh = ln.replace("(", "").replace(")", "").split()
        srows.append(f"{float(f):.1f} & {n} & {float(s_td):.2f} & {float(s_wh):.2f} \\\\")
    write("tab_tdconv_sl.tex", "\n".join(srows) + "\n")


def runs_table():
    """Every 2D configuration: TD periods/steps/s, WH iters/steps/s, CW iters/s."""
    import timing as T
    a_h, _ = args_for(HC, "run")
    a_y, _ = args_for(YC, "run")
    rows, last = [], None
    for ant, n, f, d in T.configs():
        if last is not None and ant != last:
            rows.append("\\midrule")
        last = ant
        k = tkey(ant, n, f, d)
        if ant == "horn":
            cw = HC.simulate(a_h, "cw", f, d, only_cached=True)
        else:
            cw = YC.simulate(a_y, "cw", n, f, d, only_cached=True)
        cs = W.stats_of(cw) if cw is not None else {}
        name = "horn" if ant == "horn" else "Yagi"
        rows.append(
            f"{name} & {n} & {f:g} & {d:g} & {fmt(tget('td', k, 'periods'), 1)} & "
            f"{tget('td', k, 'steps') or '--'} & {fmt(tget('td', k, 'wall_s'), 1)} & "
            f"{tget('wh', k, 'iters') or '--'} & {tget('wh', k, 'steps') or '--'} & "
            f"{fmt(tget('wh', k, 'wall_s'), 1)} & {cs.get('iters', '--')} & "
            f"{fmt(cs.get('wall_s'), 0)} \\\\")
    write("tab_runs.tex", "\n".join(rows) + "\n")


def td_transient():
    """Transient-removed TD (td_converged.py): eps history figure and sweep table."""
    import json
    import glob
    d = W.RESULTS / "td_converged"
    plt = C.pyplot()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    cmap = plt.get_cmap("viridis")
    for pth in sorted(glob.glob(str(d / "runs" / "*.npz"))):
        with np.load(pth, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            st = json.loads(str(z["stats_json"]))
        if not st.get("eps_history"):
            continue
        eps = np.array(st["eps_history"])
        per = meta["periods"] * (np.arange(len(eps)) + 2)
        ramp = meta.get("ramp", 0.0)
        if abs(meta["f_ghz"] - 3.0) < 1e-9 and (not ramp or meta["max_windows"] != 1000):
            ax.semilogy(per, eps, lw=0.8, ls="--" if not ramp else "-",
                        label=("abrupt switch-on" if not ramp else f"ramp {ramp:g} periods") +
                        f" (test, 3 GHz)")
        elif ramp == 3.0 and meta["max_windows"] == 1000 and meta["tol"] == 1e-6:
            c = cmap((meta["f_ghz"] - 0.3) / 2.7)
            ax2.semilogy(per, eps, lw=0.7, color=c, label=f"{meta['f_ghz']:g} GHz")
    for a_ in (ax, ax2):
        a_.axhline(1e-6, color="k", ls=":", lw=0.8)
        a_.set_xlabel("periods")
        a_.set_ylabel(r"$\epsilon_k = \|w_k - w_{k-1}\| / \|w_k\|$")
        a_.grid(alpha=0.3, which="both")
    ax.set_title("3 GHz: abrupt vs ramped switch-on", fontsize=9)
    ax2.set_title("Beamwidth sweep, ramp 3 periods, cap 2000 periods", fontsize=9)
    ax.legend(fontsize=7)
    ax2.legend(fontsize=6, ncol=2)
    C.save(fig, d / "eps_3GHz.png")
    plt.close(fig)

    # table from cached runs, so frequencies appear as they finish
    import td_converged as TDC
    runs = {}
    for pth in sorted(glob.glob(str(d / "runs" / "*.npz"))):
        with np.load(pth, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if (meta["tol"] == 1e-6 and meta.get("ramp") == 3.0 and meta["max_windows"] == 1000
                    and meta["dphi_e"] == 0.0):
                runs[meta["f_ghz"]] = {k: z[k] for k in z.files}

    def wh_cached(f, tol):
        a, _ = HC.parse_args(["run", "--solvers", "", "--show", "wh", "--tol", f"{tol:g}",
                              "--maxiters-wh", "5000"])
        return HC.simulate(a, "wh", f, 0.0, only_cached=True)

    rows = []
    for f in HC.BW_FREQS:
        if f not in runs:
            rows.append(f"{f:.2f} & \\multicolumn{{10}}{{l}}{{running / queued}} \\\\")
            continue
        r, st = runs[f], W.stats_of(runs[f])
        whT, wh6 = wh_cached(f, 1e-10), wh_cached(f, 1e-6)
        sT = W.stats_of(whT) if whT is not None else {}
        rows.append(
            f"{f:.2f} & {st['periods']} & {st['timesteps']} & {st['wall_s']:.0f} & "
            f"{sci(st['est_error'])} & {'yes' if st['converged'] else 'cap'} & "
            f"{sci(TDC.rel_far(r, whT)) if whT is not None else '--'} & "
            f"{sci(TDC.rel_far(wh6, whT)) if whT is not None and wh6 is not None else '--'} & "
            f"{sci(HC.pattern_diff_db(r, whT)) if whT is not None else '--'} & "
            f"{sT.get('iters', '--')} & {fmt(sT.get('wall_s'), 1)} \\\\")
    write("tab_tdtrans.tex", "\n".join(rows) + "\n")


def cost_table(cnt):
    rows = []
    td = list(TIM.get("td", {}).values())
    if td:
        per = np.array([x["periods"] for x in td])
        st = np.array([x["steps"] for x in td])
        wall = np.array([x["wall_s"] for x in td])
        rows.append(f"time domain & {len(td)} & {np.median(per):.0f} periods "
                    f"({per.min():.0f}--{per.max():.0f}) & {int(np.median(st))} & "
                    f"{np.median(wall):.1f} & {wall.sum() / 3600:.2f} & -- \\\\")
    else:
        rows.append("time domain & -- & -- & -- & -- & -- & -- \\\\")
    for s in ("wh", "cw"):
        c = cnt.cost[s]
        if not c:
            rows.append(f"{TEXLABEL[s]} & 0 & -- & -- & -- & -- & -- \\\\")
            continue
        it = np.array([x["iters"] for x in c])
        steps = np.array([x["timesteps"] for x in c])
        wall = np.array([x["wall_s"] for x in c])
        if s == "wh" and TIM.get("wh") and len(TIM["wh"]) == len(c):
            wall = np.array([x["wall_s"] for x in TIM["wh"].values()])   # idle machine
        per = np.array([x.get("periodicity", np.nan) for x in c])
        conv = sum(x["converged"] for x in c)
        rows.append(f"{TEXLABEL[s]} & {len(c)} ({conv} conv.) & {int(np.median(it))} "
                    f"({it.min()}--{it.max()}) & {int(np.median(steps))} & {np.median(wall):.1f} & "
                    f"{wall.sum() / 3600:.2f} & {sci(float(np.median(per)))} \\\\")
    write("tab_cost.tex", "\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    opt = ap.parse_args()

    # 1. figures (cached data only)
    for mod, cmd in ((HC, "bw"), (HC, "fig4"), (HC, "table1"), (HC, "fig6"),
                     (YC, "fig4"), (YC, "fig5")):
        a, fn = args_for(mod, cmd)
        fn(a)

    # 2. tables
    global TIM
    TIM = timing()
    cnt = Count()
    horn_bw(cnt)
    horn_steer(cnt)
    horn_scan(cnt)
    patterns(cnt, HC, "fig6", [(f, d) for f in (0.5, 5.0) for d in (0.0, 10.0, 20.0, 30.0)],
             "tab_horn_patterns.tex")
    patterns(cnt, YC, "fig5", [(16, f, d) for f in (2.5, 5.0) for d in (0.0, 20.0, 40.0, 60.0)],
             "tab_yagi_patterns.tex")
    yagi_steer(cnt)
    td_convergence_tables()
    runs_table()
    td_transient()
    # cost over distinct cached runs
    uniq = Count()
    import glob
    for s in ("wh", "cw"):
        for pth in sorted(glob.glob(str(W.RESULTS / "runs" / f"*_{s}_*.npz"))):
            with np.load(pth, allow_pickle=False) as z:
                if json.loads(str(z["meta"])).get("tol") != 1e-6:
                    continue          # tight-tolerance references (td_converged.py)
                uniq.cost[s].append(W.stats_of({k: z[k] for k in z.files}))
    cost_table(uniq)
    # distinct 2D configurations: horn bw 12 + fig4 6 + table1 5 + fig6 7; yagi fig4 18 + fig5 4
    n_total = 52
    write("status.tex",
          f"\\newcommand{{\\nWH}}{{{len(uniq.cost['wh'])}}}\n"
          f"\\newcommand{{\\nCW}}{{{len(uniq.cost['cw'])}}}\n"
          f"\\newcommand{{\\nTOTAL}}{{{n_total}}}\n"
          f"\\newcommand{{\\builddate}}{{{datetime.now():%Y-%m-%d %H:%M}}}\n")
    print(f"tables written to {GEN}; wh runs {len(uniq.cost['wh'])}, cw runs {len(uniq.cost['cw'])}")

    # 3. pdf
    if not opt.no_pdf:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "cw_report.tex"],
                           cwd=REPORT, stdout=subprocess.DEVNULL)
        for ext in ("aux", "log", "out"):
            (REPORT / f"cw_report.{ext}").unlink(missing_ok=True)
        print("wrote", REPORT / "cw_report.pdf")


if __name__ == "__main__":
    main()
