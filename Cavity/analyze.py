"""Collect cavity benchmark logs into a CSV, print scaling/convergence tables, and plot them.

    python analyze.py results            # every results/<mode>/*.log, mode = strong|weak|conv|...

Each log holds one run: the RESULT line from cavity.cpp / cavity.py plus MEEP's own
"Finished solve_cw after N CG iters" / "Finished solve_wh after N GMRES|BiCGSTAB iters" line.

Iteration counts are not comparable between methods: one BiCGSTAB(L) iteration applies the
operator 2L times, a GMRES iteration once. "applies" counts operator applications and "steps"
the FDTD time steps they cost (one WaveHoltz apply = periods * steps_per_period steps, one
solve_cw apply = one step), which is the cost measure that compares across solvers.
Writes <dir>/summary.csv and, if matplotlib is available, <dir>/<mode>.png.
"""
import csv
import math
import pathlib
import re
import sys
from collections import defaultdict

ITERS = re.compile(r"Finished solve_(?:cw|wh) after (\d+)")
NUMERIC = {"np", "res", "sx", "sy", "sz", "omega", "eps", "cells", "courant", "steps_per_period",
           "tol", "L", "periods", "converged", "t_setup", "t_solve", "t_error", "relerr"}


def parse(log):
    text = log.read_text(errors="replace")
    lines = [l for l in text.splitlines() if l.startswith("RESULT ")]
    if not lines:
        return None
    r = dict(kv.split("=", 1) for kv in lines[-1].split()[1:])
    for k in NUMERIC & r.keys():
        r[k] = float(r[k])
    its = ITERS.findall(text)
    r["iters"] = int(its[-1]) if its else math.nan
    krylov_bicgstab = r["solver"] == "cw" or r.get("whsolver") == "bicgstab"
    r["applies"] = r["iters"] * (2 * r["L"] if krylov_bicgstab else 1)
    r["steps"] = r["applies"] * (r["periods"] * r["steps_per_period"] if r["solver"] == "wh" else 1)
    r["t_per_step"] = r["t_solve"] / r["steps"] if r["steps"] > 0 else math.nan
    r["mode"] = log.parent.name
    r["log"] = str(log)
    return r


def series_name(r):
    if r["solver"] != "wh":
        return f"{r['iface']} {r['solver']}"
    return f"{r['iface']} wh ({r['whsolver']}, {r['periods']:g}T)"


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[series_name(r)].append(r)
    return {k: sorted(v, key=lambda r: (r["np"], r["res"])) for k, v in sorted(g.items())}


def table(title, header, body):
    print(f"\n{title}\n  " + "  ".join(f"{h:>10}" for h in header))
    for row in body:
        print("  " + "  ".join(f"{x:>10.4g}" if isinstance(x, float) else f"{x:>10}" for x in row))


def strong(rows):
    out = {}
    for name, rs in group(rows).items():
        b = rs[0]
        speed = [b["t_solve"] / r["t_solve"] * b["np"] for r in rs]  # relative to 1 rank ideal
        eff = [s / r["np"] for s, r in zip(speed, rs)]
        table(f"strong scaling: {name}  (res {b['res']:g}, {b['cells']:.3g} cells)",
              ["np", "iters", "steps", "t_solve", "t/step", "speedup", "eff"],
              [[int(r["np"]), r["iters"], r["steps"], r["t_solve"], r["t_per_step"], s, e]
               for r, s, e in zip(rs, speed, eff)])
        out[name] = ([r["np"] for r in rs], speed, eff)
    return out


def weak(rows):
    out = {}
    for name, rs in group(rows).items():
        b = rs[0]
        eff = [b["t_solve"] / r["t_solve"] for r in rs]
        eff_it = [b["t_per_step"] / r["t_per_step"] for r in rs]
        table(f"weak scaling: {name}  ({b['cells'] / b['np']:.3g} cells/rank at np {b['np']:g})",
              ["np", "cells", "iters", "steps", "t_solve", "t/step", "eff", "eff/step"],
              [[int(r["np"]), r["cells"], r["iters"], r["steps"], r["t_solve"], r["t_per_step"], e, ei]
               for r, e, ei in zip(rs, eff, eff_it)])
        out[name] = ([r["np"] for r in rs], eff, eff_it)
    return out


def conv(rows):
    out = {}
    for name, rs in group(rows).items():
        rs = sorted(rs, key=lambda r: r["res"])
        order = [math.nan] + [math.log(a["relerr"] / b["relerr"]) / math.log(b["res"] / a["res"])
                              for a, b in zip(rs, rs[1:])]
        table(f"convergence: {name}",
              ["res", "np", "iters", "applies", "steps", "t_solve", "rel L2 err", "order"],
              [[r["res"], int(r["np"]), r["iters"], r["applies"], r["steps"], r["t_solve"],
                r["relerr"], o]
               for r, o in zip(rs, order)])
        out[name] = ([r["res"] for r in rs], [r["relerr"] for r in rs])
    return out


# Color and marker identify the solver variant, line style the interface (C++ solid, Python
# dashed). Variants take categorical slots in a fixed order of first appearance, never cycled.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = "osD^vPXh"


def style(variants, name):
    i = variants.index(name.split(" ", 1)[1])
    if i >= len(COLORS):
        sys.exit(f"more than {len(COLORS)} solver variants in one plot; split the results directory")
    return dict(color=COLORS[i], marker=MARKERS[i], markersize=5, linewidth=1.5,
                linestyle="--" if name.startswith("py") else "-", label=name)


def plot(mode, data, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(data)
    variants = list(dict.fromkeys(n.split(" ", 1)[1] for n in names))
    if mode == "conv":
        fig, axes = plt.subplots(1, 1, figsize=(5, 4), squeeze=False)
        ax = axes[0, 0]
        for n in names:
            ax.loglog(*data[n], **style(variants, n))
        res, err = data[names[0]]
        ax.loglog(res, [err[0] * (res[0] / r) ** 2 for r in res], color="#999", lw=1, ls=":",
                  label="order 2")
        ax.set(xlabel="resolution (pixels / unit)", ylabel="relative L2 error",
               title="Grid convergence")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
        a, b = axes[0]
        for n in names:
            nps, y1, y2 = data[n]
            if mode == "strong":
                a.loglog(nps, y1, **style(variants, n))
            else:
                a.semilogx(nps, y1, **style(variants, n))
            b.semilogx(nps, y2, **style(variants, n))
        nps = sorted({p for n in names for p in data[n][0]})
        if mode == "strong":
            a.loglog(nps, nps, color="#999", lw=1, ls=":", label="ideal")
            a.set(ylabel="speedup (vs. 1 rank, extrapolated)", title="Strong scaling: speedup")
            b.set(ylabel="parallel efficiency", title="Strong scaling: efficiency")
        else:
            a.set(ylabel="efficiency t(1)/t(P)", title="Weak scaling: solve time")
            b.set(ylabel="efficiency per time step", title="Weak scaling: time per FDTD step")
        for ax in (a, b):
            ax.set_xlabel("MPI ranks")
        b.set_ylim(0, 1.1)
        if mode != "strong":
            a.set_ylim(0, 1.1)
    for ax in axes.flat:
        ax.grid(True, which="major", color="#e0e0e0", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = [r for r in map(parse, sorted(root.rglob("*.log"))) if r]
    if not rows:
        sys.exit(f"no RESULT lines in {root}/**/*.log")
    keys = ["mode", "iface", "solver", "whsolver", "np", "res", "sx", "sy", "sz", "cells",
            "omega", "tol", "L", "periods", "steps_per_period", "converged", "iters", "applies",
            "steps", "t_setup", "t_solve", "t_per_step", "t_error", "relerr", "log"]
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {root / 'summary.csv'} ({len(rows)} runs)")
    if any(not r["converged"] for r in rows):
        print("NOT CONVERGED: " + ", ".join(r["log"] for r in rows if not r["converged"]))

    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    for mode, rs in sorted(by_mode.items()):
        kind = mode.split("_")[0] if mode.split("_")[0] in ("strong", "weak") else "conv"
        data = {"strong": strong, "weak": weak, "conv": conv}[kind](rs)
        plot(kind, data, root / f"{mode}.png")


if __name__ == "__main__":
    main()
