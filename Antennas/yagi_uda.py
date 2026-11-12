"""Yagi-Uda phased arrays in MEEP: Hanson, Electronics 2021, 10, 415, Secs. 3-4.

Element (Fig. 3 left, Table 1), in the xy-plane, boresight +x:

    reflector (length r) -- d1 -- radiator (length y, driven) -- d1 --
    director (z) -- d1 -- director (z) -- d1 -- director (z)

joined by a boom of length L = 4 d1 along x.  The black parts are metal
(PEC); the radiator is the red line of Fig. 3, a y-directed line current with
complex amplitude exp(-1j*i*dPhi) on element i.  Elements are stacked along y
with spacing dy (and along z with dz = dy for the N x N array of Sec. 4).

  2D (Sec. 3): the elements are infinite in z, Ey polarization, E-plane pattern.
  3D (Sec. 4): wires of square cross-section, N x N array in the yz-plane,
               E-plane (xy) and H-plane (xz) patterns.

Not specified in the paper, chosen here (all overridable): wire width
0.2 cm, 4 cm of free space between the array and the near-to-far box, cell
width 42 cm along x (Fig. 3 shows x in [-21, 21]).

Reproduce the paper (results/ gets one .npz per run, cached, plus figures):

    python yagi_uda.py fig4          # Fig. 4 top: beam angle / BW vs dPhi, N = 8, 16
    python yagi_uda.py fig5          # Fig. 5: N = 16 patterns, 2.5 & 5 GHz
    python yagi_uda.py fig9          # Fig. 9: 16 x 16 array, 3D, 36 steering runs
    python yagi_uda.py fig11         # Fig. 11: 16 x 16 array, 3D, E/H patterns
    python yagi_uda.py run --n 16 --freq 5 --dphi 20      # one configuration

Any subcommand runs under MPI: mpirun -np 8 python yagi_uda.py fig11
"""

import argparse
import sys

import numpy as np
import meep as mp

import common as C

# Table 1, Yagi-Uda column (cm)
TABLE1 = dict(L=7.20, d1=1.80, r=3.75, y=2.81, z=1.24, dy=3.92, resolution=6)


def geometry(n, dim, p, wire, backplane_x=None):
    """Metal parts and radiator line sources for an n (x n) Yagi-Uda array."""
    ys = C.array_positions(n, p["dy"])
    zs = C.array_positions(n, p["dy"]) if dim == 3 else [0.0]
    x0 = -0.5 * p["L"]
    xs = {"reflector": x0, "radiator": x0 + p["d1"],
          "directors": [x0 + k * p["d1"] for k in (2, 3, 4)]}
    wz = wire if dim == 3 else mp.inf

    geom, radiators = [], []
    for j, zc in enumerate(zs):
        for i, yc in enumerate(ys):
            geom.append(mp.Block(center=mp.Vector3(0, yc, zc),
                                 size=mp.Vector3(p["L"], wire, wz), material=C.metal(p)))
            geom.append(mp.Block(center=mp.Vector3(xs["reflector"], yc, zc),
                                 size=mp.Vector3(wire, p["r"], wz), material=C.metal(p)))
            for xd in xs["directors"]:
                geom.append(mp.Block(center=mp.Vector3(xd, yc, zc),
                                     size=mp.Vector3(wire, p["z"], wz), material=C.metal(p)))
            radiators.append((i, j, mp.Vector3(xs["radiator"], yc, zc),
                              mp.Vector3(0, p["y"], 0)))
    if backplane_x is not None:
        span = (n - 1) * p["dy"] + p["r"] + 2.0
        geom.append(mp.Block(center=mp.Vector3(backplane_x, 0, 0),
                             size=mp.Vector3(0.5, span, span if dim == 3 else mp.inf),
                             material=C.metal(p)))
    return geom, radiators


def cell_size(n, dim, p, dpml, pad, sx):
    span = (n - 1) * p["dy"] + p["r"] + 2 * pad + 2 * dpml
    return mp.Vector3(sx, span, span if dim == 3 else 0)


def simulate(a, n, f_ghz, dphi_e, dphi_h=0.0, dim=2):
    """One run; returns {"angles", "P_E"[, "P_H"], "meta"} (cached on disk)."""
    p = dict(TABLE1)
    if getattr(a, "conductivity", None) is not None:
        p["conductivity"] = a.conductivity   # only set when non-default (PEC), keeps cache keys
    res = a.resolution or p["resolution"]
    until = C.default_until(f_ghz, a.until_ns, a.min_periods)
    meta = dict(antenna="yagi", dim=dim, n=n, f_ghz=f_ghz, dphi_e=dphi_e,
                dphi_h=dphi_h if dim == 3 else 0.0, resolution=res, until=until,
                dpml=a.dpml, n2f_gap=a.n2f_gap, pad=a.pad, sx=a.sx, wire=a.wire,
                backplane_x=a.backplane_x, dang=a.dang, complex=a.complex,
                subpixel=not a.no_subpixel, **{k: v for k, v in p.items() if k != "resolution"})
    path = C.cache_path(a.outdir, f"yagi{dim}d_N{n}_f{f_ghz:g}_E{dphi_e:g}_H{meta['dphi_h']:g}", meta)

    def compute():
        fcen = C.ghz(f_ghz)
        geom, rads = geometry(n, dim, p, a.wire, a.backplane_x)
        src = [mp.Source(mp.ContinuousSource(frequency=fcen, width=a.source_width),
                         component=mp.Ey, center=c, size=s,
                         amplitude=C.element_phase(i, j, dphi_e, meta["dphi_h"]))
               for i, j, c, s in rads]
        sim = mp.Simulation(cell_size=cell_size(n, dim, p, a.dpml, a.pad, a.sx),
                            resolution=res, geometry=geom, sources=src,
                            boundary_layers=[mp.PML(a.dpml)],
                            eps_averaging=not a.no_subpixel,
                            force_complex_fields=a.complex)
        out = C.run_pattern(sim, fcen, until, "EH" if dim == 3 else "E",
                            a.dang, a.dpml, a.n2f_gap)
        out["meta"] = meta
        return out

    if mp.am_master():
        print(f"== Yagi {dim}D N={n}{'x%d' % n if dim == 3 else ''} f={f_ghz} GHz "
              f"dPhiE={dphi_e} dPhiH={meta['dphi_h']}  res={res} until={until:.1f}")
    return C.cached(path, compute, a.force)


def d_over_lambda(f_ghz):
    return TABLE1["dy"] / C.wavelength_cm(f_ghz)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def cmd_run(a):
    r = simulate(a, a.n, a.freq, a.dphi, a.dphi_h, a.dim)
    if mp.am_master():
        C.plot_run(r, a.outdir / f"yagi{a.dim}d_N{a.n}_f{a.freq:g}_E{a.dphi:g}_H{a.dphi_h:g}.png")


def steering_sweep(a, n, f_ghz, dphis):
    rows = []
    for dp in dphis:
        r = simulate(a, n, f_ghz, dp, 0.0, 2)
        if mp.am_master():
            meas, th = C.report(r)
            rows.append((dp, meas["peak_deg"], meas["bw_deg"], th["peak_deg"]))
    return np.array(rows)


def cmd_fig4(a):
    """Fig. 4 top: |dphi| / BW(dPhi=0) vs dPhi at 5 GHz, N = 8 and N = 16."""
    f = 5.0
    dphis = np.arange(0, 81, 10.0)
    data = {n: steering_sweep(a, n, f, dphis) for n in (8, 16)}
    if not mp.am_master():
        return
    plt = C.pyplot()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    slope_th = C.linear_slope(d_over_lambda(f))
    for ax, n in zip(axs, (8, 16)):
        d = data[n]
        bw0 = d[0, 2]
        y = np.abs(d[:, 1]) / bw0
        fit = np.polyfit(d[:, 0], y, 1)
        xx = np.linspace(0, 90, 10)
        ax.plot(d[:, 0], y, "ks", mfc="none", label="MEEP")
        ax.plot(xx, np.polyval(fit, xx), "k-", lw=1.2,
                label=f"fit, slope {fit[0] * bw0:.4f}")
        ax.plot(xx, slope_th * xx / bw0, color="0.6", lw=1.2,
                label=f"$\\lambda/(2\\pi d_y)$ = {slope_th:.4f}")
        ax.set_xlabel(r"$\Delta\Phi$ (deg)")
        ax.set_ylabel(r"$|\Delta\phi|$ / BW")
        ax.set_title(f"N = {n}, {f:g} GHz, BW = {bw0:.1f} deg "
                     f"(theory {C.theory_metrics(n, d_over_lambda(f), 0)['bw_deg']:.1f})",
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
        bw_paper = np.rad2deg(0.886 / ((n - 1) * d_over_lambda(f)))
        print(f"N={n}: fitted slope {fit[0] * bw0:.4f} deg/deg, theory {slope_th:.4f}; "
              f"BW(0) = {bw0:.2f} deg (Eq. 25: "
              f"{C.theory_metrics(n, d_over_lambda(f), 0)['bw_deg']:.2f}, "
              f"0.886 lambda/((N-1) d): {bw_paper:.2f}, paper: {11.0 if n == 8 else 5.0})")
    C.save(fig, a.outdir / "yagi_fig4.png")


def cmd_fig5(a):
    """Fig. 5: N = 16 E-plane patterns, f = 2.5, 5 GHz, dPhi = 0, 20, 40, 60."""
    cases = [(f, dp) for f in (2.5, 5.0) for dp in (0, 20, 40, 60)]
    res = [simulate(a, 16, f, dp, 0.0, 2) for f, dp in cases]
    if not mp.am_master():
        return
    plt = C.pyplot()
    fig, axs = plt.subplots(4, 2, subplot_kw={"projection": "polar"}, figsize=(7, 13))
    for ax, (f, dp), r in zip(axs.flat, cases, res):
        C.report(r)
        C.pattern_panel(ax, r, "E", f"f = {f} GHz, $\\Delta\\Phi$ = {dp} deg")
    axs[0, 0].legend(fontsize=6, loc="lower left")
    C.save(fig, a.outdir / "yagi_fig5.png")


def cmd_fig9(a):
    """Fig. 9: 16 x 16 array at 5 GHz, dPhiE, dPhiH in 0..75 step 15 (36 runs)."""
    f, n = 5.0, 16
    steps = np.arange(0, 76, 15.0)
    grid = {}
    for dh in steps:
        for de in steps:
            r = simulate(a, n, f, de, dh, 3)
            if mp.am_master():
                grid[de, dh] = (C.beam_metrics(r["angles"], r["P_E"]),
                                C.beam_metrics(r["angles"], r["P_H"]))
    if not mp.am_master():
        return
    plt = C.pyplot()
    fig = plt.figure(figsize=(11, 6))
    axE, axH = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 3)
    axS = fig.add_subplot(1, 2, 2)
    slope = C.linear_slope(d_over_lambda(f))
    xx = np.linspace(0, 80, 10)
    for other, mk in ((0.0, "o"), (45.0, "s")):
        axE.plot(steps, [grid[de, other][0]["peak_deg"] for de in steps], "k" + mk,
                 mfc="none", label=rf"$\Delta\Phi_H$ = {other:g}")
        axH.plot(steps, [grid[other, dh][1]["peak_deg"] for dh in steps], "k" + mk,
                 mfc="none", label=rf"$\Delta\Phi_E$ = {other:g}")
    for ax, pl in ((axE, "E"), (axH, "H")):
        ax.plot(xx, [C.steer_angle(x, d_over_lambda(f)) for x in xx], "k-", lw=1)
        ax.plot(xx, -slope * xx, color="0.6", lw=1, ls="--")
        ax.set_xlabel(rf"$\Delta\Phi_{pl}$ (deg)")
        ax.set_ylabel(rf"$\Delta\phi_{pl}$ (deg)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for (de, dh), (mE, mH) in grid.items():
        axS.errorbar(mE["peak_deg"], mH["peak_deg"], xerr=0.5 * mE["bw_deg"],
                     yerr=0.5 * mH["bw_deg"], fmt="k.", ms=3, lw=0.6)
    axS.set_xlabel(r"$\Delta\phi_E$ (deg)")
    axS.set_ylabel(r"$\Delta\phi_H$ (deg)")
    axS.set_title("16 x 16 Yagi-Uda, 5 GHz", fontsize=9)
    axS.grid(alpha=0.3)
    C.save(fig, a.outdir / "yagi_fig9.png")


def cmd_fig11(a):
    """Fig. 11: 16 x 16 array, f = 3, 4 GHz, (dPhiE, dPhiH) = (0,0), (60,30), (30,60)."""
    combos = [(0, 0), (60, 30), (30, 60)]
    freqs = (3.0, 4.0)
    res = {(f, c): simulate(a, 16, f, c[0], c[1], 3) for f in freqs for c in combos}
    if not mp.am_master():
        return
    plt = C.pyplot()
    fig, axs = plt.subplots(4, 3, subplot_kw={"projection": "polar"}, figsize=(10, 13))
    for fi, f in enumerate(freqs):
        for ci, c in enumerate(combos):
            r = res[f, c]
            print(f"f={f} GHz, (dPhiE, dPhiH)={c}")
            for pi, pl in enumerate("EH"):
                C.report(r, pl)
                C.pattern_panel(axs[2 * fi + pi, ci], r, pl,
                              f"{pl}-plane, f = {f:g} GHz, {c}")
    C.save(fig, a.outdir / "yagi_fig11.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"run": cmd_run, "fig4": cmd_fig4, "fig5": cmd_fig5,
            "fig9": cmd_fig9, "fig11": cmd_fig11}
    for name in cmds:
        p = sub.add_parser(name, help=(cmds[name].__doc__ or "").split("\n")[0])
        C.add_common_args(p)
        p.add_argument("--wire", type=float, default=0.2, help="wire width (cm)")
        p.add_argument("--conductivity", type=float, default=None,
                       help="metal as a finite conductor with this MEEP D_conductivity "
                            "(1 ~ 0.27 S/m); default PEC")
        p.add_argument("--pad", type=float, default=4.0,
                       help="free space beyond the outermost elements along y/z (cm)")
        p.add_argument("--sx", type=float, default=42.0, help="cell length along x (cm)")
        p.add_argument("--backplane-x", type=float, default=None,
                       help="add a 0.5 cm PEC back-plane at this x (reflector is at -3.6)")
        p.add_argument("--source-width", type=float, default=0.0,
                       help="ContinuousSource turn-on width (paper: 0, abrupt)")
        if name == "run":
            p.add_argument("--dim", type=int, choices=(2, 3), default=2)
            p.add_argument("--n", type=int, default=16)
            p.add_argument("--freq", type=float, default=5.0, help="GHz")
            p.add_argument("--dphi", type=float, default=0.0,
                           help="phase shift per element along y, deg (E-plane)")
            p.add_argument("--dphi-h", type=float, default=0.0,
                           help="phase shift per element along z, deg (H-plane, 3D)")
    a = ap.parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=True)
    cmds[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
