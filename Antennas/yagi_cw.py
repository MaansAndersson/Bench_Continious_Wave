"""Yagi-Uda phased arrays (yagi_uda.py) solved with solve_cw and solve_wh (MeepHoltz).

Same geometry, sources, near-to-far box and far-field analysis as
yagi_uda.py; see cw_common.py for how the far field is extracted from each
solver.  Every command overlays the cached time-domain result when it exists.

    python yagi_cw.py fig4      # Fig. 4 top: steering, N = 8 and 16 at 5 GHz
    python yagi_cw.py fig5      # Fig. 5: N = 16 patterns, 2.5 and 5 GHz
    python yagi_cw.py fig9      # Fig. 9 (3D 16 x 16)
    python yagi_cw.py fig11     # Fig. 11 (3D 16 x 16)
    python yagi_cw.py run --n 16 --freq 5 --dphi 40
"""

import argparse
import sys

import numpy as np
import meep as mp

import cw_common as W
from cw_common import C
import yagi_uda as Y      # yagi_uda.py
from horn_cw import MARK, STYLE, metric, overlay, pattern_diff_db, save_rows


def simulate(a, solver, n, f_ghz, dphi_e, dphi_h=0.0, dim=2, only_cached=False):
    f_ghz, dphi_e, dphi_h = float(f_ghz), float(dphi_e), float(dphi_h)  # int vs float changes the hash
    p = dict(Y.TABLE1)
    if getattr(a, "conductivity", None) is not None:
        p["conductivity"] = a.conductivity   # only set when non-default (PEC), keeps cache keys
    res = a.resolution or p["resolution"]
    fcen = C.ghz(f_ghz)
    dh = dphi_h if dim == 3 else 0.0
    a.exact_dt = "conductivity" in p          # see cw_common.courant_for
    courant, nt = W.courant_for(fcen, res, a.periods, exact=a.exact_dt)
    meta = dict(antenna="yagi", solver=solver, dim=dim, n=n, f_ghz=f_ghz, dphi_e=dphi_e,
                dphi_h=dh, resolution=res, dpml=a.dpml, n2f_gap=a.n2f_gap, pad=a.pad,
                sx=a.sx, wire=a.wire, backplane_x=a.backplane_x, dang=a.dang,
                subpixel=not a.no_subpixel, courant=courant, tol=a.tol,
                **{k: v for k, v in p.items() if k != "resolution"})
    meta.update({"L": a.L, "farfield": "n2f-before-solve"} if solver == "cw" else
                W.wh_meta(a))
    if getattr(a, "save_fields", False):
        meta["fields"] = True                # cached run holds the Ey solution
    tag = f"yagi{dim}d_{solver}_N{n}_f{f_ghz:g}_E{dphi_e:g}_H{dh:g}"
    path = C.cache_path(a.outdir, tag, meta)
    if only_cached:
        return W.cached_or(path, None, a, True)
    if mp.am_master():
        print(f"== Yagi {dim}D [{solver}] N={n} f={f_ghz} GHz dPhi=({dphi_e}, {dh}) "
              f"res={res} Courant={courant:.4f} ({nt} steps / {a.periods} periods)")

    geom, rads = Y.geometry(n, dim, p, a.wire, a.backplane_x)
    cell = Y.cell_size(n, dim, p, a.dpml, a.pad, a.sx)
    sources = [mp.Source(mp.ContinuousSource(frequency=fcen), component=mp.Ey, center=c,
                         size=s, amplitude=C.element_phase(i, j, dphi_e, dh))
               for i, j, c, s in rads]

    def make_sim(courant_, cplx):
        return mp.Simulation(cell_size=cell, resolution=res, geometry=geom, sources=sources,
                             boundary_layers=[mp.PML(a.dpml)],
                             eps_averaging=not a.no_subpixel, Courant=courant_,
                             force_complex_fields=cplx)

    def compute():
        a.resolution_used = res
        out = W.cw_pattern(make_sim, solver, fcen, "EH" if dim == 3 else "E", a,
                           a.dpml, a.n2f_gap)
        out["meta"] = meta
        return out

    return C.cached(path, compute, a.force)


def all_results(a, n, f, de, dh=0.0, dim=2):
    res = {s: r for s, oc in W.solver_plan(a)
           if (r := simulate(a, s, n, f, de, dh, dim, oc)) is not None}
    t = W.td_result(antenna="yagi", dim=dim, n=n, f_ghz=f, dphi_e=de,
                    dphi_h=dh if dim == 3 else 0.0,
                    resolution=a.resolution or Y.TABLE1["resolution"], dpml=a.dpml)
    if t is not None:
        res = {"td": t, **res}
    elif mp.am_master():
        print(f"   (no cached time-domain result for N={n} f={f} dPhi=({de},{dh}))")
    return res


def cmd_run(a):
    res = all_results(a, a.n, a.freq, a.dphi, a.dphi_h, a.dim)
    if not mp.am_master():
        return
    plt = C.pyplot()
    planes = "EH" if a.dim == 3 else "E"
    fig, axs = plt.subplots(1, len(planes), subplot_kw={"projection": "polar"},
                            figsize=(4.5 * len(planes), 4.5), squeeze=False)
    for ax, pl in zip(axs[0], planes):
        overlay(ax, res, pl, f"{pl}-plane, N={a.n}, {a.freq:g} GHz, ({a.dphi:g}, {a.dphi_h:g})")
    axs[0, 0].legend(fontsize=6, loc="lower left")
    C.save(fig, a.outdir / f"yagi{a.dim}d_N{a.n}_f{a.freq:g}_E{a.dphi:g}_H{a.dphi_h:g}.png")


def cmd_fig4(a):
    """Fig. 4 top: beam angle and BW vs dPhi at 5 GHz, N = 8 and 16, all solvers."""
    f = 5.0
    dps = np.arange(0, 81, 10.0)
    res = {(n, d): all_results(a, n, f, d) for n in (8, 16) for d in dps}
    if not mp.am_master():
        return
    dl = Y.d_over_lambda(f)
    slope_th = C.linear_slope(dl)
    plt = C.pyplot()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    rows = []
    for ax, n in zip(axs, (8, 16)):
        names = [x for x in ("td", "cw", "wh") if any(x in res[n, d] for d in dps)]
        for s in names:
            ds = np.array([d for d in dps if s in res[n, d]])
            ang = np.array([abs(metric(res[n, d][s])["peak_deg"]) for d in ds])
            bw = np.array([metric(res[n, d][s])["bw_deg"] for d in ds])
            fit = np.polyfit(ds, ang, 1) if len(ds) >= 2 else [np.nan]
            bw0 = bw[0] if ds[0] == 0 else np.nan
            ax.plot(ds, ang / bw0, MARK[s], mfc="none", color=STYLE[s]["color"],
                    label=f"{s} ({len(ds)}/{len(dps)}): slope {fit[0]:.4f}, BW(0) {bw0:.3f}")
            rows.append(f"N={n:2d} {s} ({len(ds)} pts): slope {fit[0]:.5f} (theory {slope_th:.5f}), "
                        f"BW(0) {bw0:.4f}, |angles| " + " ".join(f"{x:.3f}" for x in ang))
        for s, ref in (("wh", "td"), ("cw", "td"), ("wh", "cw")):
            ds = [d for d in dps if s in res[n, d] and ref in res[n, d]]
            if ds:
                rows.append(f"N={n:2d} max pattern diff {s}-{ref}: "
                            f"{max(pattern_diff_db(res[n, d][s], res[n, d][ref]) for d in ds):.2e} dB")
        xx = np.linspace(0, 85, 10)
        bw0 = metric(next(iter(res[n, 0.0].values())))["bw_deg"]
        ax.plot(xx, slope_th * xx / bw0, color="0.6", lw=1, label=r"$\lambda/(2\pi d_y)$")
        ax.set_xlabel(r"$\Delta\Phi$ (deg)")
        ax.set_ylabel(r"$|\Delta\phi|$ / BW")
        ax.set_title(f"Yagi N = {n}, 5 GHz", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    C.save(fig, a.outdir / "yagi_fig4.png")
    save_rows(a, "yagi_fig4.txt", "Yagi steering at 5 GHz", rows)


def pattern_grid(a, cases, dim, fname, planes="E", ncol=2):
    res = [all_results(a, n, f, de, dh, dim) for n, f, de, dh in cases]
    if not mp.am_master():
        return
    plt = C.pyplot()
    nrow = len(cases) * len(planes) // ncol
    fig, axs = plt.subplots(nrow, ncol, subplot_kw={"projection": "polar"},
                            figsize=(3.6 * ncol, 3.5 * nrow), squeeze=False)
    panels = []
    for k in range(0, len(cases), ncol):
        for pl in planes:
            panels += [(res[q], cases[q], pl) for q in range(k, k + ncol)]
    rows = []
    for ax, (r, (n, f, de, dh), pl) in zip(axs.flat, panels):
        print(f"N={n} f={f} GHz dPhi=({de}, {dh}) {pl}-plane")
        overlay(ax, r, pl, f"{pl}-plane, {f:g} GHz, ({de:g}, {dh:g})")
        rows.append(f"{f:4.1f} ({de:g},{dh:g}) {pl}: " + "  ".join(
            f"{s} beam {metric(x, pl)['peak_deg']:8.3f} BW {metric(x, pl)['bw_deg']:7.3f}"
            for s, x in r.items()) +
            (f"  dP(wh-cw) {pattern_diff_db(r['wh'], r['cw'], pl):.1e}" if "wh" in r and "cw" in r else "") +
            (f"  dP(cw-td) {pattern_diff_db(r['cw'], r['td'], pl):.2f}" if "td" in r and "cw" in r else ""))
    axs[0, 0].legend(fontsize=6, loc="lower left")
    C.save(fig, a.outdir / fname)
    save_rows(a, fname.replace(".png", ".txt"), "patterns", rows)


def cmd_fig5(a):
    """Fig. 5: N = 16 patterns at 2.5 and 5 GHz, dPhi = 0, 20, 40, 60."""
    pattern_grid(a, [(16, f, d, 0.0) for f in (2.5, 5.0) for d in (0, 20, 40, 60)], 2,
                 "yagi_fig5.png")


def cmd_fig9(a):
    """Fig. 9 (3D 16 x 16): steering grid at 5 GHz."""
    steps = np.arange(0, 76, 15.0)
    for dh in steps:
        for de in steps:
            all_results(a, 16, 5.0, de, dh, 3)


def cmd_fig11(a):
    """Fig. 11 (3D 16 x 16): E/H patterns at 3 and 4 GHz."""
    cases = [(16, f, de, dh) for f in (3.0, 4.0) for de, dh in ((0, 0), (60, 30), (30, 60))]
    pattern_grid(a, cases, 3, "yagi_fig11.png", planes="EH", ncol=3)


def parse_args(argv):
    """(namespace, command function) for argv, e.g. ["bw", "--solvers", ""]."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"run": cmd_run, "fig4": cmd_fig4, "fig5": cmd_fig5, "fig9": cmd_fig9,
            "fig11": cmd_fig11}
    for name, fn in cmds.items():
        p = sub.add_parser(name, help=(fn.__doc__ or "").split("\n")[0])
        C.add_common_args(p)
        W.add_solver_args(p)
        p.set_defaults(outdir=W.RESULTS)
        p.add_argument("--wire", type=float, default=0.2)
        p.add_argument("--pad", type=float, default=4.0)
        p.add_argument("--sx", type=float, default=42.0)
        p.add_argument("--backplane-x", type=float, default=None)
        if name == "run":
            p.add_argument("--dim", type=int, choices=(2, 3), default=2)
            p.add_argument("--n", type=int, default=16)
            p.add_argument("--freq", type=float, default=5.0)
            p.add_argument("--dphi", type=float, default=0.0)
            p.add_argument("--dphi-h", type=float, default=0.0)
    a = ap.parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=True)
    return a, cmds[a.cmd]


def main(argv=None):
    a, cmd = parse_args(argv)
    cmd(a)


if __name__ == "__main__":
    main(sys.argv[1:])
