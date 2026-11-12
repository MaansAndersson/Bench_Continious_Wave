"""Horn array beamwidth vs. frequency at a reduced resolution: solve_cw and solve_wh.

The table "Horn array: beamwidth vs. frequency" of report/cw_report.tex, recomputed with the
two frequency-domain solvers only (no time domain) at RES pixels/cm instead of the paper's 6,
so that the whole sweep (12 frequencies x 2 solvers) finishes within an hour on this machine.
The horn wall is one pixel thick (WALL = 1/RES cm): the paper's 0.2 cm would be thinner than a
pixel, which wall_transparency.py shows can leak.

    python fast_bw.py run            # all runs, NJOBS in parallel, cached in results/cw/fast_bw
    python fast_bw.py report         # tables, figures, reports/fast_bw/fast_bw_report.pdf
    python fast_bw.py job SOLVER F   # one run (used by `run`)

Every run keeps its Ey solution (--save-fields): results/cw/fast_bw/runs/horn2d_{cw,wh}_*.npz
hold Ey (complex, whole cell), eps, x, y next to the far field.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import cw_common as W
from cw_common import C
import horn as H
import horn_cw as HC

RES = 3
WALL = 1.0 / RES
CW_L = 10                         # BiCGSTAB(L) of solve_cw
FREQS = HC.BW_FREQS
FIELD_F = 1.5                     # frequency of the field figure
OUT = W.RESULTS / "fast_bw"
REPORT = W.REPORTS / "fast_bw"
GEN = REPORT / "generated"
TIMING = OUT / "timing.json"
NJOBS = os.cpu_count() or 4
SOLVERS = ("cw", "wh")
LABEL = {"cw": f"solve_cw (BiCGSTAB-{CW_L})", "wh": "solve_wh (GMRES(50), 2 periods)"}


def cw_args(solver=""):
    """horn_cw options of this study; solver '' only reads cached runs."""
    a, _ = HC.parse_args(["bw", "--solvers", solver, "--show", "" if solver else "cw,wh",
                          "--resolution", f"{RES:g}", "--wall", f"{WALL:.10g}", "--L", str(CW_L),
                          "--outdir", str(OUT), "--save-fields"])
    return a


def results(f):
    """{"cw", "wh"} for one frequency, cached runs only."""
    a = cw_args()
    return {s: r for s in SOLVERS
            if (r := HC.simulate(a, s, f, 0.0, only_cached=True)) is not None}


def old_run(solver, f, res=6, tol=1e-6):
    """A run of the old report (cw_report, results/runs, 6 px/cm, wall 0.2 cm) or None.

    Old wh runs were made with solve_waveholtz_cw (no "impl" key), so they are found by
    scanning the metadata rather than through HC.simulate.
    """
    import glob
    for fn in sorted(glob.glob(str(W.RESULTS / "runs" / f"horn2d_{solver}_N16_f{f:g}_E0_H0_*.npz"))):
        with np.load(fn, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
        m = json.loads(str(out["meta"]))
        if m["resolution"] == res and m["tol"] == tol and "impl" not in m and "fields" not in m:
            out["meta"] = m
            return out
    return None


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def cmd_job(a):
    solver, f = a.solver, float(a.freq)
    cached = HC.simulate(cw_args(), solver, f, 0.0, only_cached=True) is not None
    t0 = time.time()
    HC.simulate(cw_args(solver), solver, f, 0.0)
    print(f"JOBTIME {solver} {f:g} {time.time() - t0:.2f}{' CACHED' if cached else ''}")


def cmd_run(a):
    """Run every (solver, f) not cached yet, NJOBS at a time, cw (the slowest) first."""
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [(s, f) for s in SOLVERS for f in FREQS]
    db = json.loads(TIMING.read_text()) if TIMING.exists() else {"jobs": {}}
    t0 = time.time()
    running = {}
    while jobs or running:
        while jobs and len(running) < a.njobs:
            s, f = jobs.pop(0)
            log = open(OUT / "logs" / f"{s}_f{f:g}.log", "w")
            p = subprocess.Popen([sys.executable, __file__, "job", s, f"{f:g}"], cwd=W.HERE,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[p] = (s, f, log, time.time())
        time.sleep(1)
        for p in [p for p in running if p.poll() is not None]:
            s, f, log, ts = running.pop(p)
            log.close()
            txt = (OUT / "logs" / f"{s}_f{f:g}.log").read_text()
            ok = p.returncode == 0 and "JOBTIME" in txt
            if ok and "CACHED" not in txt:
                db["jobs"][f"{s}_f{f:g}"] = dict(wall_s=time.time() - ts, njobs=a.njobs)
            print(f"[{time.time() - t0:7.0f} s] {s} f={f:g} {'done' if ok else 'FAILED'} "
                  f"({time.time() - ts:.0f} s)", flush=True)
    if a.record_sweep:
        db["sweep_wall_s"] = time.time() - t0
    db.update(res=RES, wall=WALL, njobs=a.njobs)
    TIMING.write_text(json.dumps(db, indent=1, sort_keys=True))
    print(f"sweep finished in {time.time() - t0:.0f} s")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def sci(x):
    if x is None or not np.isfinite(x):
        return "--"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def bw(r, s):
    return HC.metric(r[s])["bw_deg"] if s in r else None


def dP(r):
    return HC.pattern_diff_db(r["wh"], r["cw"]) if "wh" in r and "cw" in r else None


def field_fit(r):
    """Ey of wh and cw on a common complex scale; returns (wh, cw, rel. max difference)."""
    a, b = r["wh"]["Ey"], r["cw"]["Ey"]
    b = b * np.vdot(b, a) / np.vdot(b, b)
    return a, b, float(np.max(np.abs(a - b)) / np.max(np.abs(a)))


def bw_figure(data, th, fn):
    plt = C.pyplot()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    axs[0].plot(FREQS, [th[f] for f in FREQS], color="r", lw=0.8, label="Eq. (25)")
    for s in SOLVERS:
        fs = [f for f in FREQS if s in data[f]]
        if not fs:
            continue
        b = [bw(data[f], s) for f in fs]
        lab = LABEL[s]
        if len(fs) >= 3:
            (ka, kb), (sa, sb) = C.fit_inverse_f(fs, b)
            lab += f": a = {ka:.2f}$\\pm${sa:.2f}, b = {kb:.2f}$\\pm${sb:.2f}"
        axs[0].plot(fs, b, HC.MARK[s], mfc="none", color=HC.STYLE[s]["color"], label=lab)
        st = [W.stats_of(data[f][s]) for f in fs]
        axs[2].semilogy(fs, [x["timesteps"] for x in st], HC.MARK[s] + "-", mfc="none",
                        color=HC.STYLE[s]["color"], label=LABEL[s])
    fs = [f for f in FREQS if dP(data[f]) is not None]
    axs[1].semilogy(fs, [max(dP(data[f]), 1e-8) for f in fs], "kx-", label="far field, dB")
    axs[1].semilogy(fs, [field_fit(data[f])[2] for f in fs], "o-", mfc="none",
                    color="tab:green", label="$E_y$ in the cell, relative")
    for ax, (xl, yl, t) in zip(axs, [
            ("Frequency (GHz)", "Beam width (deg)", f"N = 16 horns, {RES:g} px/cm; fit a/f + b"),
            ("Frequency (GHz)", "max difference", "solve_wh vs solve_cw"),
            ("Frequency (GHz)", "time steps (operator cost)", "Cost in time steps")]):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    C.save(fig, fn)
    plt.close(fig)


def field_figure(r, fn):
    plt = C.pyplot()
    from matplotlib.colors import ListedColormap
    wh = r["wh"]["Ey"]
    cw = field_fit(r)[1] if "cw" in r else None
    z = r["wh"]
    s = np.exp(-1j * np.angle(wh.flat[np.argmax(np.abs(wh))]))    # real at the peak
    x, y, eps = z["x"], z["y"], z["eps"]
    ext = [x[0], x[-1], y[0], y[-1]]
    metal = np.ma.masked_where(np.abs(eps - 1) < 1e-6, np.ones_like(eps))
    vmax = np.percentile(np.abs(wh), 99.5)
    amax = np.abs(wh).max()
    panels = [((wh * s).real, "RdBu_r", (-vmax, vmax), "Re $E_y$ (solve_wh)"),
              (20 * np.log10(np.maximum(np.abs(wh), 1e-300) / amax), "inferno", (-50, 0),
               "$|E_y|$ (dB, solve_wh)"),
              ] + ([(np.log10(np.maximum(np.abs(wh - cw) / amax, 1e-12)), "viridis", (-9, -4),
                     "$\\log_{10}|E_y^{wh}-E_y^{cw}|/\\max|E_y|$")] if cw is not None else [])
    fig, axs = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 7.6))
    for ax, (d, cmap, (lo, hi), title) in zip(axs, panels):
        im = ax.imshow(d.T, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi,
                       interpolation="nearest")
        ax.imshow(metal.T, origin="lower", extent=ext, cmap=ListedColormap(["k"]),
                  interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (cm)")
        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03)
    axs[0].set_ylabel("y (cm)")
    C.save(fig, fn)
    plt.close(fig)


def cmd_report(a):
    GEN.mkdir(parents=True, exist_ok=True)
    tim = json.loads(TIMING.read_text()) if TIMING.exists() else {"jobs": {}}
    g = cw_args()
    p = H.params(2, g)
    data = {f: results(f) for f in FREQS}
    th = {f: C.theory_metrics(p["n"], H.dl(p, f), 0)["bw_deg"] for f in FREQS}

    rows, txt = [], ["f_GHz  theory   wh      cw      [iters, steps, solve s]"]
    for f, r in data.items():
        st = {s: W.stats_of(r[s]) if s in r else {} for s in SOLVERS}
        ferr = field_fit(r)[2] if len(r) == 2 else None
        for x in st.values():                        # dagger: stopped before the tolerance
            if x and not x["converged"]:
                x["iters"] = f"{x['iters']}$^\\dagger$"
        rows.append(
            f"{f:.2f} & {th[f]:.2f} & {fmt(bw(r, 'wh'), 3)} & {fmt(bw(r, 'cw'), 3)} & "
            f"{st['wh'].get('iters', '--')} & {st['wh'].get('timesteps', '--')} & "
            f"{fmt(st['wh'].get('wall_s'), 1)} & "
            f"{st['cw'].get('iters', '--')} & {st['cw'].get('timesteps', '--')} & "
            f"{fmt(st['cw'].get('wall_s'), 0)} & {sci(dP(r))} & {sci(ferr)} \\\\")
        txt.append(f"{f:5.2f} {th[f]:7.3f} {fmt(bw(r, 'wh'))} {fmt(bw(r, 'cw'))}  "
                   + "  ".join(f"{s} [{st[s].get('iters')}, {st[s].get('timesteps')}, "
                               f"{fmt(st[s].get('wall_s'), 1)}]" for s in SOLVERS))
    (GEN / "tab_bw.tex").write_text("\n".join(rows) + "\n")
    (OUT / "horn_bw.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))

    label = {"wh": r"\texttt{solve\_wh}", "cw": r"\texttt{solve\_cw}"}
    fits = []
    for s in ("wh", "cw"):
        fs = [f for f in FREQS if s in data[f]]
        if len(fs) >= 3:
            (ka, kb), (sa, sb) = C.fit_inverse_f(fs, [bw(data[f], s) for f in fs])
            fits.append(f"{label[s]}, {RES:g} px/cm & {len(fs)} & ${ka:.2f}\\pm{sa:.2f}$ & "
                        f"${kb:.2f}\\pm{sb:.2f}$ \\\\")
        else:
            fits.append(f"{label[s]}, {RES:g} px/cm & {len(fs)} & -- & -- \\\\")
    fits.append(r"\midrule")
    fits.append(r"WaveHoltz, 6 px/cm, wall 0.2 cm (\texttt{cw\_report}) & 12 & "
                r"$11.43\pm0.19$ & $0.69\pm0.24$ \\")
    fits.append(r"paper (Hanson 2021) & 12 & $12.0\pm0.1$ & $1.1\pm0.2$ \\")
    (GEN / "tab_bw_fit.tex").write_text("\n".join(fits) + "\n")

    ref = []                       # old report, 6 px/cm, frequencies with both solvers
    for f in FREQS:
        r6 = {s: x for s in SOLVERS if (x := old_run(s, f)) is not None}
        if len(r6) == 2:
            st6 = {s: W.stats_of(r6[s]) for s in SOLVERS}
            ref.append(f"{f:.2f} & {bw(r6, 'wh'):.3f} & {bw(r6, 'cw'):.3f} & "
                       f"{st6['wh']['iters']} & {st6['wh']['timesteps']} & "
                       f"{st6['wh']['wall_s']:.1f} & {st6['cw']['iters']} & "
                       f"{st6['cw']['timesteps']} & {st6['cw']['wall_s']:.0f} & {sci(dP(r6))} \\\\")
    (GEN / "tab_bw_old.tex").write_text("\n".join(ref) + "\n")

    bw_figure(data, th, OUT / "horn_bw.png")
    rf = data[FIELD_F]
    field_figure(rf, OUT / "horn_field.png")

    jt = lambda s: sum(v["wall_s"] for k, v in tim["jobs"].items() if k.startswith(s + "_"))  # noqa: E731
    macros = {
        "builddate": time.strftime("%Y-%m-%d %H:%M"),
        "res": f"{RES:g}", "wall": f"{WALL:.3f}",
        "nx": len(rf["wh"]["x"]), "ny": len(rf["wh"]["y"]),
        "njobs": tim.get("njobs", NJOBS),
        "sweepmin": fmt(tim.get("sweep_wall_s", float("nan")) / 60, 1),
        "cwmin": fmt(jt("cw") / 60, 1), "whmin": fmt(jt("wh") / 60, 1),
        "nF": len(FREQS), "fieldf": f"{FIELD_F:g}", "fielderr": sci(field_fit(rf)[2]),
        "tol": sci(g.tol), "cwL": CW_L,
        "meepholtz": subprocess.run(["git", "-C", os.path.expanduser("~/Develop/MeepHoltz"),
                                     "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True).stdout.strip() or "?",
    }
    (GEN / "status.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n"
                                            for k, v in macros.items()))
    if not a.no_pdf:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "fast_bw_report.tex"],
                           cwd=REPORT, stdout=subprocess.DEVNULL)
        print("wrote", REPORT / "fast_bw_report.pdf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--njobs", type=int, default=NJOBS)
    p.add_argument("--record-sweep", action="store_true",
                   help="store this sweep's wall time as the one reported")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--no-pdf", action="store_true")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("field", help="only the field figure (results/cw/fast_bw/horn_field.png)")
    p.set_defaults(fn=lambda a: field_figure(results(FIELD_F), OUT / "horn_field.png"))
    p = sub.add_parser("job")
    p.add_argument("solver", choices=SOLVERS)
    p.add_argument("freq")
    p.set_defaults(fn=cmd_job)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
