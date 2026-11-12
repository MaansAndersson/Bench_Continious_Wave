"""Horn phased arrays (horn.py) solved with solve_cw and solve_wh (MeepHoltz).

Same geometry, sources, near-to-far box and far-field analysis as horn.py;
the time-harmonic field comes from a frequency-domain solve instead of time
stepping (see cw_common.py).  Every command overlays the cached time-domain
result of horn.py when it exists.

    python horn_cw.py bw        # Fig. 4 bottom right: beamwidth vs frequency, N = 16
    python horn_cw.py fig4      # Fig. 4 bottom left: steering at 0.3/1.5/3 GHz
    python horn_cw.py table1    # Table 1: scan loss at dPhi = 80 deg
    python horn_cw.py fig6      # Fig. 6: patterns at 0.5 and 5 GHz
    python horn_cw.py fig10     # Fig. 10 (3D, 8 x 8), not run by default anywhere
    python horn_cw.py fig12     # Fig. 12 (3D, 8 x 8)
    python horn_cw.py run --freq 2.5 --dphi -35 --solvers cw,wh
"""

import argparse
import json
import sys

import numpy as np
import meep as mp

import cw_common as W
from cw_common import C
import horn as H          # horn.py 


def simulate(a, solver, f_ghz, dphi_e, dphi_h=0.0, dim=2, only_cached=False):
    """One frequency-domain run; {"angles", "P_E"[, "P_H"], "stats_json", "meta"}."""
    f_ghz, dphi_e, dphi_h = float(f_ghz), float(dphi_e), float(dphi_h)  # int vs float changes the hash
    p = H.params(dim, a)
    fcen = C.ghz(f_ghz)
    dh = dphi_h if dim == 3 else 0.0
    a.exact_dt = "conductivity" in p          # see cw_common.courant_for
    courant, nt = W.courant_for(fcen, p["resolution"], a.periods, exact=a.exact_dt)
    meta = dict(antenna="horn", solver=solver, dim=dim, f_ghz=f_ghz, dphi_e=dphi_e,
                dphi_h=dh, dpml=a.dpml, n2f_gap=a.n2f_gap, sx=a.sx, sy=a.sy, x0=a.x0,
                dang=a.dang, subpixel=not a.no_subpixel, courant=courant, tol=a.tol, **p)
    meta.update({"L": a.L, "farfield": "n2f-before-solve"} if solver == "cw" else
                W.wh_meta(a))
    if getattr(a, "save_fields", False):
        meta["fields"] = True                # cached run holds the Ey solution
    tag = f"horn{dim}d_{solver}_N{p['n']}_f{f_ghz:g}_E{dphi_e:g}_H{dh:g}"
    path = C.cache_path(a.outdir, tag, meta)
    if only_cached:
        return W.cached_or(path, None, a, True)
    if mp.am_master():
        print(f"== Horn {dim}D [{solver}] N={p['n']} f={f_ghz} GHz dPhi=({dphi_e}, {dh}) "
              f"res={p['resolution']} Courant={courant:.4f} ({nt} steps / {a.periods} periods)")

    x0, cell = H.geometry_setup(a, dim, p)
    geom, rads = H.layout(p, dim, x0)
    sources = [mp.Source(mp.ContinuousSource(frequency=fcen), component=mp.Ey, center=c,
                         size=s, amplitude=C.element_phase(i, j, dphi_e, dh))
               for i, j, c, s in rads]

    def make_sim(courant_, cplx):
        return mp.Simulation(cell_size=cell, resolution=p["resolution"], geometry=geom,
                             sources=sources, boundary_layers=[mp.PML(a.dpml)],
                             eps_averaging=not a.no_subpixel, Courant=courant_,
                             force_complex_fields=cplx)

    def compute():
        a.resolution_used = p["resolution"]
        out = W.cw_pattern(make_sim, solver, fcen, "EH" if dim == 3 else "E", a,
                           a.dpml, a.n2f_gap)
        out["meta"] = meta
        return out

    return C.cached(path, compute, a.force)


def td(a, f_ghz, dphi_e, dphi_h=0.0, dim=2):
    """The horn.py time-domain result for the same case (or None)."""
    p = H.params(dim, a)
    return W.td_result(antenna="horn", dim=dim, f_ghz=f_ghz, dphi_e=dphi_e,
                       dphi_h=dphi_h if dim == 3 else 0.0, n=p["n"],
                       resolution=p["resolution"], wall=p["wall"], dpml=a.dpml)


def solvers(a):
    return [s for s in a.solvers.split(",") if s]


def all_results(a, f, de, dh=0.0, dim=2):
    """{"td": ..., "cw": ..., "wh": ...} for one case (td only if cached)."""
    res = {s: r for s, oc in W.solver_plan(a)
           if (r := simulate(a, s, f, de, dh, dim, oc)) is not None}
    t = td(a, f, de, dh, dim)
    if t is not None:
        res = {"td": t, **res}
    elif mp.am_master():
        print(f"   (no cached time-domain result for f={f}, dPhi=({de},{dh}); "
              f"run horn.py first to include it)")
    return res


STYLE = {"td": dict(color="0.25", ls="-", lw=1.4), "cw": dict(color="tab:blue", ls="--", lw=1.1),
         "wh": dict(color="tab:orange", ls=":", lw=1.4)}
MARK = {"td": "o", "cw": "s", "wh": "^"}


def metric(r, plane="E"):
    """common.beam_metrics with its parabolic peak refinement clamped to one grid step.

    On a flat or monotonic pattern (e.g. 0.5 GHz, dPhi = 80 deg, where the requested
    steering is outside visible space) the parent's refinement divides by a near-zero
    curvature and can return an angle outside the search window.
    """
    ang, p = r["angles"], r["P_" + plane]
    m = C.beam_metrics(ang, p)
    inwin = (ang >= -90.0) & (ang <= 90.0)
    grid_peak = ang[np.flatnonzero(inwin)[np.argmax(p[inwin])]]
    if abs(m["peak_deg"] - grid_peak) > ang[1] - ang[0]:
        m["peak_deg"] = float(grid_peak)
    return m


def pattern_diff_db(r, ref, plane="E", floor=-30.0):
    """Max |dB difference| between two normalised patterns where either is above floor."""
    def db(x):
        return 10 * np.log10(np.maximum(x["P_" + plane], 1e-300) / metric(x, plane)["peak_power"])
    d1, d2 = db(r), db(ref)
    m = (d1 > floor) | (d2 > floor)
    return float(np.max(np.abs(np.clip(d1, floor, 0) - np.clip(d2, floor, 0))[m]))


def save_rows(a, name, header, rows):
    txt = header + "\n" + "\n".join(rows) + "\n"
    (a.outdir / name).write_text(txt)
    print(txt)
    print("wrote", a.outdir / name)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_run(a):
    res = all_results(a, a.freq, a.dphi, a.dphi_h, a.dim)
    if not mp.am_master():
        return
    plt = C.pyplot()
    planes = "EH" if a.dim == 3 else "E"
    fig, axs = plt.subplots(1, len(planes), subplot_kw={"projection": "polar"},
                            figsize=(4.5 * len(planes), 4.5), squeeze=False)
    for ax, pl in zip(axs[0], planes):
        overlay(ax, res, pl, f"{pl}-plane, {a.freq:g} GHz, ({a.dphi:g}, {a.dphi_h:g})")
    axs[0, 0].legend(fontsize=6, loc="lower left")
    C.save(fig, a.outdir / f"horn{a.dim}d_f{a.freq:g}_E{a.dphi:g}_H{a.dphi_h:g}.png")


def overlay(ax, res, plane, title, rmin=-40.0):
    first = next(iter(res.values()))
    m = first["meta"]
    dphi = m["dphi_e"] if plane == "E" else m["dphi_h"]
    th = C.array_factor(first["angles"], m["n"], C.d_over_lambda(m), dphi)
    ang = np.deg2rad(np.append(first["angles"], first["angles"][0]))
    th_db = np.clip(10 * np.log10(np.maximum(th, 1e-300)), rmin, 0)
    ax.plot(ang, np.append(th_db, th_db[0]), color="r", lw=0.8, label="Eq. (25)")
    for s, r in res.items():
        db = np.clip(10 * np.log10(np.maximum(r["P_" + plane], 1e-300) /
                                   metric(r, plane)["peak_power"]), rmin, 0)
        ax.plot(ang, np.append(db, db[0]), label=s, **STYLE[s])
        print(f"   {s}: " + ", ".join(f"{k} {v:.3f}" for k, v in metric(r, plane).items()
                                      if k != "peak_power"))
    ax.set_ylim(rmin, 0)
    ax.set_yticks([-30, -15, -3, 0])
    ax.set_yticklabels(["-30", "-15", "-3", "0"], fontsize=7)
    ax.set_thetagrids(range(0, 360, 45), fontsize=7)
    ax.set_title(title, fontsize=8)


BW_FREQS = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]


def have(res, keys, s):
    """Keys of `res` (dict of per-case solver dicts) for which solver s is available."""
    return [k for k in keys if s in res[k]]


def cmd_bw(a):
    """Fig. 4 bottom right: unsteered beamwidth vs frequency (N = 16), all solvers."""
    freqs = BW_FREQS
    res = {f: all_results(a, f, 0.0) for f in freqs}
    if not mp.am_master():
        return
    p = H.params(2, a)
    names = [s for s in ("td", "cw", "wh") if have(res, freqs, s)]
    plt = C.pyplot()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    th = {f: C.theory_metrics(p["n"], H.dl(p, f), 0)["bw_deg"] for f in freqs}
    axs[0].plot(freqs, [th[f] for f in freqs], color="r", lw=0.8, label="Eq. (25)")
    fits = {}
    for s in names:
        fs = have(res, freqs, s)
        bw = [metric(res[f][s])["bw_deg"] for f in fs]
        lab = f"{s} ({len(fs)}/{len(freqs)} f)"
        if len(fs) >= 3:
            (ka, kb), (sa, sb) = C.fit_inverse_f(fs, bw)
            fits[s] = (ka, sa, kb, sb, len(fs))
            lab += f": a = {ka:.2f}$\\pm${sa:.2f}, b = {kb:.2f}$\\pm${sb:.2f}"
        axs[0].plot(fs, bw, MARK[s], mfc="none", color=STYLE[s]["color"], label=lab)
    axs[0].set_xlabel("Frequency (GHz)")
    axs[0].set_ylabel("Beam width (deg)")
    axs[0].set_title("N = 16 horns, $\\Delta\\Phi$ = 0; fit a/f + b", fontsize=9)
    axs[0].legend(fontsize=7)
    axs[0].grid(alpha=0.3)
    pairs = [(x, "td") for x in ("cw", "wh")] + [("wh", "cw")]
    for s, ref in pairs:
        fs = [f for f in freqs if s in res[f] and ref in res[f]]
        if fs:
            axs[1].semilogy(fs, [max(pattern_diff_db(res[f][s], res[f][ref]), 1e-6) for f in fs],
                            (MARK[s] + "-") if ref == "td" else "kx-", mfc="none",
                            color=STYLE[s]["color"] if ref == "td" else "k", label=f"{s} vs {ref}")
    axs[1].set_xlabel("Frequency (GHz)")
    axs[1].set_ylabel("max |$\\Delta$P| above -30 dB (dB)")
    axs[1].set_title("Pattern differences", fontsize=9)
    axs[1].legend(fontsize=7)
    axs[1].grid(alpha=0.3, which="both")
    for s in [x for x in names if x != "td"]:
        fs = have(res, freqs, s)
        axs[2].semilogy(fs, [W.stats_of(res[f][s])["timesteps"] for f in fs],
                        MARK[s] + "-", mfc="none", color=STYLE[s]["color"], label=s)
    axs[2].semilogy(freqs, [C.default_until(f, 6.67, 10) * p["resolution"] / 0.5 for f in freqs],
                    "o-", mfc="none", color=STYLE["td"]["color"], label="td (run length)")
    axs[2].set_xlabel("Frequency (GHz)")
    axs[2].set_ylabel("time steps (operator cost)")
    axs[2].set_title("Cost in time steps", fontsize=9)
    axs[2].legend(fontsize=7)
    axs[2].grid(alpha=0.3, which="both")
    C.save(fig, a.outdir / "horn_bw.png")

    rows = []
    for f in freqs:
        row = f"{f:5.2f}  theory {th[f]:7.3f}"
        for s in names:
            if s in res[f]:
                row += f"  {s} {metric(res[f][s])['bw_deg']:7.3f}"
                if s != "td":
                    st = W.stats_of(res[f][s])
                    row += f" [{st['iters']} it, {st['timesteps']} steps, {st['wall_s']:.1f} s]"
        rows.append(row)
    rows += [f"fit {s} ({v[4]} f): a = {v[0]:.3f} +- {v[1]:.3f}, b = {v[2]:.3f} +- {v[3]:.3f}"
             for s, v in fits.items()]
    save_rows(a, "horn_bw.txt", "beamwidth (deg) vs frequency, N = 16, dPhi = 0", rows)


def cmd_fig4(a):
    """Fig. 4 bottom left: steering at 0.3, 1.5, 3 GHz, dPhi = 0, 10, 20."""
    p = H.params(2, a)
    fs, dps = (0.3, 1.5, 3.0), (0.0, 10.0, 20.0)
    res = {(f, d): all_results(a, f, d) for f in fs for d in dps}
    if not mp.am_master():
        return
    rows = []
    for f in fs:
        for d in dps:
            r = res[f, d]
            rows.append(f"{f:4.1f} {d:5.1f} {C.steer_angle(d, H.dl(p, f)):8.3f} " +
                        " ".join(f"{s}:{metric(r[s])['peak_deg']:8.3f}" for s in r))
    save_rows(a, "horn_fig4_steering.txt", "f_GHz dPhi theory  beam angles (deg)", rows)


def cmd_table1(a):
    """Table 1: scan loss P_peak(80) - P_peak(0) at 0.5, 1, 2, 4 GHz."""
    rows = []
    for f in (0.5, 1.0, 2.0, 4.0):
        r0, r80 = all_results(a, f, 0.0), all_results(a, f, 80.0)
        if mp.am_master():
            rows.append(f"{f:4.1f}  " + "  ".join(
                f"{s}:{10 * np.log10(metric(r80[s])['peak_power'] / metric(r0[s])['peak_power']):7.2f}"
                for s in r0 if s in r80))
    if mp.am_master():
        save_rows(a, "horn_table1.txt", "f_GHz  scan loss SL_dB at dPhi = 80 (paper: "
                  "-11.6, -1.2, -1.0, -0.9)", rows)


def pattern_grid(a, cases, dim, fname, planes="E", ncol=2):
    res = [all_results(a, f, de, dh, dim) for f, de, dh in cases]
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
    for ax, (r, (f, de, dh), pl) in zip(axs.flat, panels):
        print(f"f={f} GHz, dPhi=({de}, {dh}), {pl}-plane")
        overlay(ax, r, pl, f"{pl}-plane, {f:g} GHz, ({de:g}, {dh:g})")
    axs[0, 0].legend(fontsize=6, loc="lower left")
    C.save(fig, a.outdir / fname)


def cmd_fig6(a):
    """Fig. 6: N = 16 patterns, 0.5 and 5 GHz, dPhi = 0, 10, 20, 30."""
    pattern_grid(a, [(f, d, 0.0) for f in (0.5, 5.0) for d in (0, 10, 20, 30)], 2,
                 "horn_fig6.png")


def cmd_fig10(a):
    """Fig. 10 (3D 8 x 8): beamwidth vs frequency, E and H planes."""
    freqs = [round(0.1 * k, 1) for k in range(1, 13)]
    res = {f: all_results(a, f, 0.0, 0.0, 3) for f in freqs}
    if mp.am_master():
        rows = [f"{f:4.1f} " + " ".join(f"{s}:E {metric(r, 'E')['bw_deg']:6.2f} H "
                                        f"{metric(r, 'H')['bw_deg']:6.2f}" for s, r in res[f].items())
                for f in freqs]
        save_rows(a, "horn_fig10_bw.txt", "f_GHz beamwidths (deg)", rows)


def cmd_fig12(a):
    """Fig. 12 (3D 8 x 8): E/H patterns, 0.5 and 1 GHz, (0,0), (60,30), (30,60)."""
    cases = [(f, de, dh) for f in (0.5, 1.0) for de, dh in ((0, 0), (60, 30), (30, 60))]
    pattern_grid(a, cases, 3, "horn_fig12.png", planes="EH", ncol=3)


def parse_args(argv):
    """(namespace, command function) for argv, e.g. ["bw", "--solvers", ""]."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"run": cmd_run, "bw": cmd_bw, "fig4": cmd_fig4, "table1": cmd_table1,
            "fig6": cmd_fig6, "fig10": cmd_fig10, "fig12": cmd_fig12}
    for name, fn in cmds.items():
        p = sub.add_parser(name, help=(fn.__doc__ or "").split("\n")[0])
        C.add_common_args(p)
        W.add_solver_args(p)
        p.set_defaults(outdir=W.RESULTS)
        for opt, kw in (("--n", dict(type=int)), ("--wall", dict(type=float)),
                        ("--pad", dict(type=float)), ("--sx", dict(type=float)),
                        ("--sy", dict(type=float)), ("--x0", dict(type=float))):
            p.add_argument(opt, default=None, **kw)
        p.add_argument("--no-backplane", action="store_true")
        if name == "run":
            p.add_argument("--dim", type=int, choices=(2, 3), default=2)
            p.add_argument("--freq", type=float, default=2.5)
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
