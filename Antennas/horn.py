"""Horn-antenna phased arrays in MEEP: Hanson, Electronics 2021, 10, 415, Secs. 3-4.

Element (Fig. 3 right), boresight +x, origin at the centre of the back edge
of the box:

  * a box of side a: back wall at x = 0, walls at y = +-a/2 for 0 <= x <= a;
  * the flare  y = +-f(x),  f(x) = k1 exp(k2 (x - a)),  a <= x <= a + c,
    k1 = a/2, k2 = ln(2d/a)/c, so the mouth is 2d wide; built, as in the paper,
    from n = c/dx straight slices (here: thin blocks rotated along each chord);
  * the radiator: a y-directed line current across the box at x = a/2, with
    complex amplitude exp(-1j*(i*dPhiE + j*dPhiH)) on element (i, j).

Elements are stacked along y with dy = 2d (mouths touching).

  2D (Sec. 3, Table 1):  a = 0.95, c = 15, d = 3.8, dx = 0.1, N = 16, res 6.
  3D (Sec. 4, Table 2):  a = 2, c = 15, d = 8, dx = 0.5, dy = dz = 16, N = 8 x 8,
      res 4, PEC back-plane 142 x 142 x 0.5 at x = -2a.  All walls are extruded
      over a height a in z ("the width in the z-direction ... remains fixed at
      a"): the horn is two flared plates, so the Ey field between them is a TEM
      parallel-plate mode with no cutoff (plates at z = +-a/2 would give a 7.5 GHz
      TE10 cutoff, incompatible with the paper's 0.1-1.2 GHz band).

Not specified in the paper, chosen here (all overridable): wall thickness
(0.2 cm in 2D, 0.25 cm = 1 pixel in 3D), PEC walls, a line radiator spanning
the box interior, 6 cm (2D) / 3 cm (3D) of free space around the structure
inside the near-to-far box, cell width 40 cm along x (Fig. 3).

Reproduce the paper (results/ gets one .npz per run, cached, plus figures):

    python horn.py fig4       # Fig. 4 bottom: steering at 0.3/1.5/3 GHz, BW vs f
    python horn.py table1     # Table 1: scan loss at dPhi = 80 deg, 0.5-4 GHz
    python horn.py fig6       # Fig. 6: N = 16 patterns, 0.5 & 5 GHz
    python horn.py fig7       # Fig. 7: Ey snapshots 0.5-2 ns at 2.5 GHz + pattern
    python horn.py fig10      # Fig. 10: 8 x 8 array, 3D, steering + BW vs f
    python horn.py fig12      # Fig. 12: 8 x 8 array, 3D, E/H patterns
    python horn.py run --freq 2.5 --dphi -35        # one configuration

Any subcommand runs under MPI: mpirun -np 8 python horn.py fig12
"""

import argparse
import sys

import numpy as np
import meep as mp

import common as C

TABLE1 = dict(n=16, a=0.95, c=15.0, d=3.8, dx=0.1, resolution=6, wall=0.2, pad=6.0,
              backplane=None)
TABLE2 = dict(n=8, a=2.0, c=15.0, d=8.0, dx=0.5, resolution=4, wall=0.25, pad=3.0,
              backplane=dict(x=-4.0, thickness=0.5, size=142.0))     # x = -2a


def params(dim, a):
    p = dict(TABLE1 if dim == 2 else TABLE2)
    for k in ("n", "wall", "pad", "resolution"):
        if getattr(a, k, None) is not None:
            p[k] = getattr(a, k)
    if a.no_backplane:
        p["backplane"] = None
    if getattr(a, "conductivity", None) is not None:
        p["conductivity"] = a.conductivity   # only set when non-default (PEC), keeps cache keys
    if getattr(a, "radiator", "line") == "point":
        p["radiator"] = "point"      # only set when non-default, so cached line runs keep their key
    p["dy"] = 2 * p["d"]
    return p


def horn_walls(p, x0, yc, zc, dim):
    """Metal blocks of one horn whose back edge is centred at (x0, yc, zc)."""
    a, c, d, t = p["a"], p["c"], p["d"], p["wall"]
    h = a if dim == 3 else mp.inf                 # extent in z
    k1, k2 = a / 2, np.log(2 * d / a) / c
    blocks = [mp.Block(center=mp.Vector3(x0, yc, zc), size=mp.Vector3(t, a + t, h),
                       material=C.metal(p))]
    for s in (+1, -1):
        blocks.append(mp.Block(center=mp.Vector3(x0 + a / 2, yc + s * a / 2, zc),
                               size=mp.Vector3(a, t, h), material=C.metal(p)))
    nslice = int(round(c / p["dx"]))
    xs = a + np.linspace(0, c, nslice + 1)
    ys = k1 * np.exp(k2 * (xs - a))
    for s in (+1, -1):
        for k in range(nslice):
            dxk, dyk = xs[k + 1] - xs[k], s * (ys[k + 1] - ys[k])
            ln = np.hypot(dxk, dyk)
            e1 = mp.Vector3(dxk / ln, dyk / ln)
            blocks.append(mp.Block(
                center=mp.Vector3(x0 + 0.5 * (xs[k] + xs[k + 1]),
                                  yc + s * 0.5 * (ys[k] + ys[k + 1]), zc),
                size=mp.Vector3(ln + t, t, h), e1=e1, e2=mp.Vector3(-e1.y, e1.x),
                e3=mp.Vector3(0, 0, 1), material=C.metal(p)))
    return blocks


def layout(p, dim, x0):
    ys = C.array_positions(p["n"], p["dy"])
    zs = C.array_positions(p["n"], p["dy"]) if dim == 3 else [0.0]
    geom, rads = [], []
    for j, zc in enumerate(zs):
        for i, yc in enumerate(ys):
            geom += horn_walls(p, x0, yc, zc, dim)
            # line current across the box interior (default), or a point dipole at its centre
            size = 0.0 if p.get("radiator") == "point" else p["a"] - p["wall"]
            rads.append((i, j, mp.Vector3(x0 + p["a"] / 2, yc, zc), mp.Vector3(0, size, 0)))
    bp = p["backplane"]
    if bp is not None:
        geom.append(mp.Block(center=mp.Vector3(x0 + bp["x"], 0, 0),
                             size=mp.Vector3(bp["thickness"], bp["size"],
                                             bp["size"] if dim == 3 else mp.inf),
                             material=C.metal(p)))
    return geom, rads


def extent(p, dim):
    """(x_min, x_max, half-span in y/z) of the structure, relative to x0 = 0."""
    bp = p["backplane"]
    xmin = min(-p["wall"], (bp["x"] - bp["thickness"]) if bp else 0.0)
    half = 0.5 * ((p["n"] - 1) * p["dy"] + 2 * p["d"])
    if bp:
        half = max(half, 0.5 * bp["size"])
    return xmin, p["a"] + p["c"], half


def geometry_setup(a, dim, p):
    xmin, xmax, half = extent(p, dim)
    x0 = a.x0 if a.x0 is not None else -0.5 * (xmin + xmax)
    span = 2 * (half + p["pad"] + a.dpml)
    cell = mp.Vector3(a.sx if a.sx else 40.0, a.sy if a.sy else span,
                      span if dim == 3 else 0)
    return x0, cell


def make_sim(a, dim, p, f_ghz, dphi_e, dphi_h):
    x0, cell = geometry_setup(a, dim, p)
    geom, rads = layout(p, dim, x0)
    fcen = C.ghz(f_ghz)
    src = [mp.Source(mp.ContinuousSource(frequency=fcen, width=a.source_width),
                     component=mp.Ey, center=cc, size=s,
                     amplitude=C.element_phase(i, j, dphi_e, dphi_h))
           for i, j, cc, s in rads]
    if mp.am_master():
        print(f"   cell {cell.x:g} x {cell.y:g} x {cell.z:g} cm, {len(geom)} metal objects")
    return mp.Simulation(cell_size=cell, resolution=p["resolution"], geometry=geom,
                         sources=src, boundary_layers=[mp.PML(a.dpml)],
                         eps_averaging=not a.no_subpixel,
                         force_complex_fields=a.complex), x0


def run_meta(a, dim, p, f_ghz, dphi_e, dphi_h, until):
    return dict(antenna="horn", dim=dim, f_ghz=f_ghz, dphi_e=dphi_e,
                dphi_h=dphi_h if dim == 3 else 0.0, until=until, dpml=a.dpml,
                n2f_gap=a.n2f_gap, sx=a.sx, sy=a.sy, x0=a.x0, dang=a.dang,
                complex=a.complex, subpixel=not a.no_subpixel,
                source_width=a.source_width, **p)


def simulate(a, f_ghz, dphi_e, dphi_h=0.0, dim=2):
    """One run; returns {"angles", "P_E"[, "P_H"], "meta"} (cached on disk)."""
    p = params(dim, a)
    until = C.default_until(f_ghz, a.until_ns, a.min_periods)
    meta = run_meta(a, dim, p, f_ghz, dphi_e, dphi_h, until)
    tag = f"horn{dim}d_N{p['n']}_f{f_ghz:g}_E{dphi_e:g}_H{meta['dphi_h']:g}"
    if mp.am_master():
        print(f"== Horn {dim}D N={p['n']}{'x%d' % p['n'] if dim == 3 else ''} "
              f"f={f_ghz} GHz dPhiE={dphi_e} dPhiH={meta['dphi_h']}  "
              f"res={p['resolution']} until={until:.1f}")

    def compute():
        sim, _ = make_sim(a, dim, p, f_ghz, dphi_e, meta["dphi_h"])
        out = C.run_pattern(sim, C.ghz(f_ghz), until, "EH" if dim == 3 else "E",
                            a.dang, a.dpml, a.n2f_gap)
        out["meta"] = meta
        return out

    return C.cached(C.cache_path(a.outdir, tag, meta), compute, a.force)


def dl(p, f_ghz):
    return p["dy"] / C.wavelength_cm(f_ghz)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def cmd_run(a):
    r = simulate(a, a.freq, a.dphi, a.dphi_h, a.dim)
    if mp.am_master():
        C.plot_run(r, a.outdir / f"horn{a.dim}d_f{a.freq:g}_E{a.dphi:g}_H{a.dphi_h:g}.png")


def bw_vs_f(ax, freqs, bws, label, marker="o"):
    (ka, kb), (sa, sb) = C.fit_inverse_f(freqs, bws)
    ff = np.linspace(min(freqs) * 0.9, max(freqs) * 1.02, 200)
    ax.plot(freqs, bws, "k" + marker, mfc="none", label=f"MEEP {label}")
    ax.plot(ff, ka / ff + kb, "k-", lw=0.8,
            label=f"a/f + b: a = {ka:.1f}$\\pm${sa:.1f} deg GHz, b = {kb:.1f}$\\pm${sb:.1f}")
    print(f"{label}: BW = a/f + b, a = {ka:.2f} +- {sa:.2f} deg GHz, b = {kb:.2f} +- {sb:.2f} deg")


def cmd_fig4(a):
    """Fig. 4 bottom: N = 16 steering at 0.3/1.5/3 GHz, and beamwidth vs frequency."""
    p = params(2, a)
    fig_f, dphis = (0.3, 1.5, 3.0), (0.0, 10.0, 20.0)
    steer = {(f, dp): simulate(a, f, dp) for f in fig_f for dp in dphis}
    bw_f = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    bws = [simulate(a, f, 0.0) for f in bw_f]
    if not mp.am_master():
        return
    plt = C.pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for f, mk in zip(fig_f, "o+x"):
        ang = [abs(C.beam_metrics(steer[f, dp]["angles"], steer[f, dp]["P_E"])["peak_deg"])
               for dp in dphis]
        xx = np.linspace(0, 22, 50)
        th = [abs(C.steer_angle(x, dl(p, f))) for x in xx]
        ax1.plot(dphis, ang, "k" + mk, mfc="none", label=f"{f} GHz")
        ax1.plot(xx, th, color="0.6", lw=0.8)
        print(f"{f} GHz: |beam angle| {np.round(ang, 2)} deg for dPhi {dphis}; theory "
              f"{[round(abs(C.steer_angle(x, dl(p, f))), 2) for x in dphis]}")
    ax1.set_xlabel(r"$\Delta\Phi$ (deg)")
    ax1.set_ylabel(r"$|\Delta\phi|$ (deg)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    bw = [C.beam_metrics(r["angles"], r["P_E"])["bw_deg"] for r in bws]
    bw_vs_f(ax2, bw_f, bw, "N=16")
    ax2.plot(bw_f, [C.theory_metrics(p["n"], dl(p, f), 0)["bw_deg"] for f in bw_f],
             color="0.6", ls="--", lw=0.8, label="Eq. (25)")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Beam width (deg)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    C.save(fig, a.outdir / "horn_fig4.png")


def cmd_table1(a):
    """Table 1 (right): scan loss SL = P_peak(dPhi = 80) - P_peak(0), N = 16."""
    p = params(2, a)
    rows = []
    for f in (0.5, 1.0, 2.0, 4.0):
        r0, r80 = simulate(a, f, 0.0), simulate(a, f, 80.0)
        if mp.am_master():
            m0 = C.beam_metrics(r0["angles"], r0["P_E"])
            m80 = C.beam_metrics(r80["angles"], r80["P_E"])
            rows.append((f, dl(p, f), 10 * np.log10(m80["peak_power"] / m0["peak_power"]),
                         m80["peak_deg"]))
    if mp.am_master():
        paper = {0.5: -11.6, 1.0: -1.2, 2.0: -1.0, 4.0: -0.9}
        lines = ["f (GHz)  dPhi  dy/lambda  SL_dB (MEEP)  SL_dB (paper)  beam (deg)"]
        for f, d, sl, pk in rows:
            lines.append(f"{f:7.1f}  {80:4d}  {d:9.3f}  {sl:12.2f}  {paper[f]:13.1f}  {pk:10.2f}")
        txt = "\n".join(lines)
        print(txt)
        (a.outdir / "horn_table1.txt").write_text(txt + "\n")
        print("wrote", a.outdir / "horn_table1.txt")


def pattern_grid(a, cases, dim, fname, planes="E", ncol=2):
    res = [simulate(a, f, de, dh, dim) for f, de, dh in cases]
    if not mp.am_master():
        return
    plt = C.pyplot()
    nrow = len(cases) * len(planes) // ncol
    fig, axs = plt.subplots(nrow, ncol, subplot_kw={"projection": "polar"},
                            figsize=(3.4 * ncol, 3.3 * nrow), squeeze=False)
    panels = [(r, c, pl) for r, c in zip(res, cases) for pl in planes]
    if len(planes) == 2:          # paper layout: E row then H row per frequency
        panels = []
        for k in range(0, len(cases), ncol):
            for pl in planes:
                panels += [(res[q], cases[q], pl) for q in range(k, k + ncol)]
    for ax, (r, (f, de, dh), pl) in zip(axs.flat, panels):
        print(f"f={f} GHz, dPhi=({de}, {dh})")
        C.report(r, pl)
        title = f"{pl}-plane, {f:g} GHz, " + (f"({de:g}, {dh:g})" if dim == 3 else
                                              f"$\\Delta\\Phi$ = {de:g}")
        C.pattern_panel(ax, r, pl, title)
    axs[0, 0].legend(fontsize=6, loc="lower left")
    C.save(fig, a.outdir / fname)


def cmd_fig6(a):
    """Fig. 6: N = 16 E-plane patterns, f = 0.5, 5 GHz, dPhi = 0, 10, 20, 30."""
    cases = [(f, dp, 0.0) for f in (0.5, 5.0) for dp in (0, 10, 20, 30)]
    pattern_grid(a, cases, 2, "horn_fig6.png")


def cmd_fig7(a):
    """Fig. 7: Ey at t = 0.5, 1, 1.5, 2 ns, 2.5 GHz, dPhi = -35, 80 x 150 cm cell."""
    a.sx = a.sx or 80.0
    a.sy = a.sy or 150.0
    a.x0 = a.x0 if a.x0 is not None else -32.0
    f, dphi = 2.5, -35.0
    p = params(2, a)
    until = C.default_until(f, a.until_ns, a.min_periods)
    meta = run_meta(a, 2, p, f, dphi, 0.0, until)
    snap_ns = (0.5, 1.0, 1.5, 2.0)

    def compute():
        sim, _ = make_sim(a, 2, p, f, dphi, 0.0)
        n2f = sim.add_near2far(C.ghz(f), 0, 1,
                               *C.near2far_box(sim.cell_size, a.dpml, a.n2f_gap))
        snaps = []
        for t in snap_ns:
            sim.run(until=C.ns(t) - sim.meep_time())
            snaps.append(np.real(sim.get_array(component=mp.Ey)))
        sim.run(until=until - sim.meep_time())
        out = C.sim_farfield(sim, n2f, "E", a.dang)
        out["snaps"] = np.array(snaps)
        out["meta"] = meta
        sim.reset_meep()
        return out

    r = C.cached(C.cache_path(a.outdir, "horn2d_fig7", meta), compute, a.force)
    if not mp.am_master():
        return
    C.report(r)
    plt = C.pyplot()
    fig = plt.figure(figsize=(11, 9))
    fig.subplots_adjust(hspace=0.35)
    ext = [-a.sx / 2, a.sx / 2, -a.sy / 2, a.sy / 2]
    for k, (t, s) in enumerate(zip(snap_ns, r["snaps"])):
        ax = fig.add_subplot(2, 4, k + 1)
        im = ax.imshow(s.T, origin="lower", extent=ext, cmap="cool", vmin=-0.002, vmax=0.002)
        ax.set_title(f"t = {t} ns", fontsize=9)
        ax.set_xlabel("x (cm)", fontsize=8)
        if k == 0:
            ax.set_ylabel("y (cm)", fontsize=8)
    fig.colorbar(im, ax=fig.axes[:4], fraction=0.02, label="Ey")
    ax = fig.add_subplot(2, 1, 2, projection="polar")
    C.pattern_panel(ax, r, "E", f"N = 16 horns, {f} GHz, $\\Delta\\Phi$ = {dphi:g} deg")
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(a.outdir / "horn_fig7.png", dpi=150)
    print("wrote", a.outdir / "horn_fig7.png")


def cmd_fig10(a):
    """Fig. 10: 8 x 8 array, 3D: steering at 1 GHz and beamwidth vs frequency."""
    p = params(3, a)
    f = 1.0
    steps = np.arange(0, 76, 15.0)
    cases = sorted({(de, o) for de in steps for o in (0.0, 45.0)} |
                   {(o, dh) for dh in steps for o in (0.0, 45.0)})
    res = {c: simulate(a, f, c[0], c[1], 3) for c in cases}
    bw_f = [round(0.1 * k, 1) for k in range(1, 13)]
    bwr = [simulate(a, fb, 0.0, 0.0, 3) for fb in bw_f]
    if not mp.am_master():
        return
    plt = C.pyplot()
    fig = plt.figure(figsize=(11, 6))
    axE, axH, axB = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 3), fig.add_subplot(1, 2, 2)
    xx = np.linspace(0, 80, 50)
    for other, mk in ((0.0, "o"), (45.0, "s")):
        axE.plot(steps, [C.beam_metrics(res[de, other]["angles"], res[de, other]["P_E"])
                         ["peak_deg"] for de in steps], "k" + mk, mfc="none",
                 label=rf"$\Delta\Phi_H$ = {other:g}")
        axH.plot(steps, [C.beam_metrics(res[other, dh]["angles"], res[other, dh]["P_H"])
                         ["peak_deg"] for dh in steps], "k" + mk, mfc="none",
                 label=rf"$\Delta\Phi_E$ = {other:g}")
    for ax, pl in ((axE, "E"), (axH, "H")):
        ax.plot(xx, [C.steer_angle(x, dl(p, f)) for x in xx], "k-", lw=1)
        ax.set_xlabel(rf"$\Delta\Phi_{pl}$ (deg)")
        ax.set_ylabel(rf"$\Delta\phi_{pl}$ (deg)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for pl, mk in (("E", "o"), ("H", "s")):
        bw = [C.beam_metrics(r["angles"], r["P_" + pl])["bw_deg"] for r in bwr]
        bw_vs_f(axB, bw_f, bw, pl + "-plane", mk)
    axB.set_xlabel("Frequency (GHz)")
    axB.set_ylabel("Beam width (deg)")
    axB.legend(fontsize=6)
    axB.grid(alpha=0.3)
    C.save(fig, a.outdir / "horn_fig10.png")


def cmd_fig12(a):
    """Fig. 12: 8 x 8 array, 3D, f = 0.5, 1 GHz, (dPhiE, dPhiH) = (0,0), (60,30), (30,60)."""
    cases = [(f, de, dh) for f in (0.5, 1.0) for de, dh in ((0, 0), (60, 30), (30, 60))]
    pattern_grid(a, cases, 3, "horn_fig12.png", planes="EH", ncol=3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"run": cmd_run, "fig4": cmd_fig4, "table1": cmd_table1, "fig6": cmd_fig6,
            "fig7": cmd_fig7, "fig10": cmd_fig10, "fig12": cmd_fig12}
    for name in cmds:
        p = sub.add_parser(name, help=(cmds[name].__doc__ or "").split("\n")[0])
        C.add_common_args(p)
        p.add_argument("--n", type=int, default=None,
                       help="elements per array dimension (paper: 16 in 2D, 8 in 3D)")
        p.add_argument("--wall", type=float, default=None, help="metal wall thickness (cm)")
        p.add_argument("--pad", type=float, default=None,
                       help="free space around the structure along y/z (cm)")
        p.add_argument("--sx", type=float, default=None, help="cell length along x (cm)")
        p.add_argument("--sy", type=float, default=None, help="cell length along y (cm)")
        p.add_argument("--x0", type=float, default=None,
                       help="x of the horn back edge (default: centre the structure)")
        p.add_argument("--no-backplane", action="store_true",
                       help="drop the Table 2 back-plane in 3D")
        p.add_argument("--conductivity", type=float, default=None,
                       help="metal as a finite conductor with this MEEP D_conductivity "
                            "(1 ~ 0.27 S/m); default PEC")
        p.add_argument("--radiator", choices=("line", "point"), default="line",
                       help="horn radiator: y-directed line current across the box (default) "
                            "or a point dipole at the box centre")
        p.add_argument("--source-width", type=float, default=0.0,
                       help="ContinuousSource turn-on width (paper: 0, abrupt)")
        if name == "run":
            p.add_argument("--dim", type=int, choices=(2, 3), default=2)
            p.add_argument("--freq", type=float, default=2.5, help="GHz")
            p.add_argument("--dphi", type=float, default=0.0,
                           help="phase shift per element along y, deg (E-plane)")
            p.add_argument("--dphi-h", type=float, default=0.0,
                           help="phase shift per element along z, deg (H-plane, 3D)")
    a = ap.parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=True)
    cmds[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
