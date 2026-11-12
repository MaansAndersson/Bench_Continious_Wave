"""Geometry figures in the style of Fig. 3 of Hanson (2021), drawn with MEEP's plot2D.

As in MEEP's python/examples/antenna_pec_ground_plane.py: build the Simulation,
add the near-to-far box, and call sim.plot2D, which draws the metal (black), the
sources (red), the PML (green, hatched) and the Near2FarRegion box (blue).
The geometry comes from yagi_uda.py / horn.py unchanged, so the pictures are
exactly what is simulated.

    python plot_geometry.py      # results/geometry_{fig3,3d}_{yagi,horn}.{png,pdf}
"""

import argparse

import numpy as np
import meep as mp

import common as C
import horn as H
import yagi_uda as Y

DPML, N2F_GAP = 1.0, 0.5
F_PLOT = C.ghz(5.0)            # any frequency: plot2D only draws the sources' extent
# white vacuum, black metal: MEEP's plot2D draws PEC (eps = -inf) in pec_color, not the colormap
EPS = dict(cmap="binary", pec_color="black", alpha=1.0, interpolation="none")
MON = dict(color="blue", alpha=0.9, linewidth=1.6, hatch=None)
SRC = dict(color="red", alpha=1.0, linewidth=2.0, hatch=None)
# plotting resolutions (pixels/cm); plotting only needs the geometry, not the simulation grid
RES = dict(element=100, array2d=30, section=30)
DPI = 250
BND = dict(hatch="/", linewidth=1.0, edgecolor="green", facecolor="none", alpha=0.6)


def horn_args(**kw):
    a = argparse.Namespace(n=None, wall=None, pad=None, resolution=None, no_backplane=False,
                           x0=None, dpml=DPML, sx=None, sy=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def sources(rads, comp=mp.Ey):
    return [mp.Source(mp.ContinuousSource(frequency=F_PLOT), component=comp, center=c, size=s)
            for i, j, c, s in rads]


def yagi_sim(n, dim=2, resolution=6, n2f=True):
    p = dict(Y.TABLE1)
    geom, rads = Y.geometry(n, dim, p, 0.2)
    cell = Y.cell_size(n, dim, p, DPML, 4.0, 42.0)
    sim = mp.Simulation(cell_size=cell, resolution=resolution, geometry=geom,
                        sources=sources(rads), boundary_layers=[mp.PML(DPML)])
    if n2f:
        sim.add_near2far(F_PLOT, 0, 1, *C.near2far_box(cell, DPML, N2F_GAP))
    return sim, p


def horn_sim(dim=2, resolution=None, n2f=True, **kw):
    a = horn_args(**kw)
    p = H.params(dim, a)
    x0, cell = H.geometry_setup(a, dim, p)
    geom, rads = H.layout(p, dim, x0)
    sim = mp.Simulation(cell_size=cell, resolution=resolution or p["resolution"], geometry=geom,
                        sources=sources(rads), boundary_layers=[mp.PML(DPML)])
    if n2f:
        sim.add_near2far(F_PLOT, 0, 1, *C.near2far_box(cell, DPML, N2F_GAP))
    return sim, p, x0


def cross_section(sim3d, axis, value, resolution):
    """2D Simulation of the plane `axis` = value of a 3D sim, for drawing only.

    plot2D on a 3D Simulation initialises the whole 3D structure, so a resolution that
    resolves 0.2 cm wires would need ~1e8 voxels just to draw a picture. Instead, the
    objects that cross the plane are copied into a 2D cell, whose x-y axes are the plane's
    in-plane axes: (x, y) for axis "z", (y, z) for axis "x". Blocks rotated about z (the
    horn flare) are only valid for axis "z"; none crosses the radiator plane (axis "x").
    """
    def inplane(v):
        return mp.Vector3(v.x, v.y) if axis == "z" else mp.Vector3(v.y, v.z)

    def crosses(c, half):
        return abs(getattr(c, axis) - value) <= half + 1e-9

    geom = []
    for g in sim3d.geometry:
        e1, e2 = g.e1, g.e2
        rotated = abs(e1.x - 1) > 1e-12 or abs(e2.y - 1) > 1e-12
        if axis == "z":
            if not crosses(g.center, 0.5 * g.size.z):
                continue
            geom.append(mp.Block(center=inplane(g.center), material=g.material,
                                 size=mp.Vector3(g.size.x, g.size.y, mp.inf),
                                 e1=mp.Vector3(e1.x, e1.y), e2=mp.Vector3(e2.x, e2.y)))
        else:
            half_x = abs(e1.x) * 0.5 * g.size.x + abs(e2.x) * 0.5 * g.size.y
            if not crosses(g.center, half_x):
                continue
            assert not rotated, "rotated block crosses a yz cross-section"
            geom.append(mp.Block(center=inplane(g.center), material=g.material,
                                 size=mp.Vector3(g.size.y, g.size.z, mp.inf)))
    srcs = [mp.Source(s.src, component=mp.Ey, center=inplane(s.center),
                      size=mp.Vector3(*([s.size.x, s.size.y] if axis == "z" else [s.size.y, s.size.z])))
            for s in sim3d.sources if crosses(s.center, 0.5 * getattr(s.size, axis))]
    cell = inplane(sim3d.cell_size)
    sec = mp.Simulation(cell_size=cell, resolution=resolution, geometry=geom, sources=srcs,
                        boundary_layers=[mp.PML(DPML)])
    sec.add_near2far(F_PLOT, 0, 1, *C.near2far_box(cell, DPML, N2F_GAP))
    return sec


def save(fig, out):
    """out (PNG) and the same figure as PDF next to it (vector text and lines; the
    metal/source map from plot2D stays an embedded image at the plotting resolution)."""
    for fn in (out, out.with_suffix(".pdf")):
        fig.savefig(fn, dpi=DPI, bbox_inches="tight")
        print("wrote", fn)
    C.pyplot().close(fig)


def plot(sim, ax, plane=None):
    sim.plot2D(ax=ax, output_plane=plane, eps_parameters=EPS, monitor_parameters=MON,
               source_parameters=SRC, boundary_parameters=BND)


def dim_arrow(ax, p0, p1, text, offset=(0, 0), **kw):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", lw=0.8, color="k",
                                                       shrinkA=0, shrinkB=0))
    mid = (0.5 * (p0[0] + p1[0]) + offset[0], 0.5 * (p0[1] + p1[1]) + offset[1])
    ax.text(*mid, text, ha="center", va="center", fontsize=11, **kw)


def yagi_element(ax):
    """Single Yagi-Uda element with the dimensions of Fig. 3 / Table 1."""
    sim, p = yagi_sim(1, resolution=RES["element"], n2f=False)
    plot(sim, ax, mp.Volume(center=mp.Vector3(0.4, 0), size=mp.Vector3(11, 6.4)))
    L, d1, r, y, z = p["L"], p["d1"], p["r"], p["y"], p["z"]
    x0 = -L / 2
    top = r / 2 + 0.45
    for k in range(4):
        dim_arrow(ax, (x0 + k * d1, top), (x0 + (k + 1) * d1, top), "$d_1$", (0, 0.3))
    dim_arrow(ax, (x0, -r / 2 - 0.45), (x0 + L, -r / 2 - 0.45), "$L$", (0, -0.3))
    dim_arrow(ax, (x0 - 0.5, -r / 2), (x0 - 0.5, r / 2), "$r$", (-0.35, 0))
    dim_arrow(ax, (x0 + L + 0.6, -y / 2), (x0 + L + 0.6, y / 2), "$y$", (0.35, 0))
    dim_arrow(ax, (x0 + L + 0.25, -z / 2), (x0 + L + 0.25, z / 2), "", (0, 0))
    ax.text(x0 + L + 0.05, z / 2 + 0.25, "$z$", fontsize=11, ha="right")
    ax.set_title("Yagi-Uda element (Table 1): reflector, driven element (red),\n"
                 "3 directors, boom", fontsize=10)


def horn_element(ax):
    """Single horn with the dimensions of Fig. 3 / Table 1."""
    sim, p, x0 = horn_sim(resolution=RES["element"], n2f=False, n=1, sx=26.0, sy=16.0)
    plot(sim, ax, mp.Volume(center=mp.Vector3(x0 + 8.0, -0.8), size=mp.Vector3(21, 12.4)))
    a, c, d = p["a"], p["c"], p["d"]
    dim_arrow(ax, (x0, a / 2 + 0.6), (x0 + a, a / 2 + 0.6), "$a$", (0, 0.65))
    dim_arrow(ax, (x0 - 0.6, -a / 2), (x0 - 0.6, a / 2), "$a$", (-0.5, 0))
    dim_arrow(ax, (x0 + a, d + 0.7), (x0 + a + c, d + 0.7), "$c$", (0, 0.45))
    dim_arrow(ax, (x0 + a + c + 0.6, -d), (x0 + a + c + 0.6, d), "$2d$", (0.8, 0))
    ax.text(x0 - 1.4, -d - 0.6, r"$f(x) = k_1\,e^{k_2 (x-a)}$,  $k_1 = a/2$,  $k_2 = \ln(2d/a)/c$"
            "\n" fr"$n = c/dx = {int(round(c / p['dx']))}$ slices; origin at the back of the box",
            fontsize=10, va="top")
    ax.set_title("Horn element (Table 1): box, exponential flare, radiator (red)", fontsize=10)


def fig3(out, kind):
    """Fig. 3 for one antenna type: the element (top) and the N = 16 array in its cell (bottom)."""
    plt = C.pyplot()
    fig = plt.figure(figsize=(5.8, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.6], hspace=0.18)
    ax = fig.add_subplot(gs[1])
    if kind == "yagi":
        yagi_element(fig.add_subplot(gs[0]))
        sim, _ = yagi_sim(16, resolution=RES["array2d"])
        plot(sim, ax)
        ax.set_title("N = 16 Yagi-Uda array, $d_y$ = 3.92 cm", fontsize=11)
    else:
        horn_element(fig.add_subplot(gs[0]))
        sim, _, _ = horn_sim(n=16, resolution=RES["array2d"])
        plot(sim, ax)
        ax.set_title("N = 16 horn array, $d_y$ = 2d = 7.6 cm", fontsize=11)
    for a_ in fig.axes:
        a_.set_xlabel("x (cm)", fontsize=10)
        a_.set_ylabel("y (cm)", fontsize=10)
        a_.tick_params(labelsize=9)
    save(fig, out)


def fig_3d(out, kind):
    """Cross-sections of one 3D array (Sec. 4): xy-plane through a row, yz-plane at the radiators."""
    plt = C.pyplot()
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    # the 3D sims are never initialised: only their geometry, sources and cell are used
    if kind == "yagi":
        sim, p = yagi_sim(16, dim=3)
        zr = C.array_positions(16, p["dy"])[0]
        xr = -p["L"] / 2 + p["d1"]
        plot(cross_section(sim, "z", zr, RES["section"]), axs[0])
        axs[0].set_title(f"16 x 16 Yagi-Uda: xy-plane through the row at z = {zr:.2f} cm",
                         fontsize=10)
        plot(cross_section(sim, "x", xr, RES["section"]), axs[1])
        axs[1].set_title(f"16 x 16 Yagi-Uda: yz-plane at x = {xr:.2f} cm (driven elements)",
                         fontsize=10)
    else:
        sim, p, x0 = horn_sim(dim=3)
        zr = C.array_positions(p["n"], p["dy"])[0]
        plot(cross_section(sim, "z", zr, RES["section"]), axs[0])
        axs[0].set_title(f"8 x 8 horn xy-plane through the row at z = {zr:.1f} cm;\n"
                         "back-plane at x = -2a", fontsize=10)
        plot(cross_section(sim, "x", x0 + p["a"] / 2, RES["section"]), axs[1])
        axs[1].set_title(f"8 x 8 horn: yz-plane at x = {x0 + p['a'] / 2:.1f} cm (radiators)",
                         fontsize=10)
    axs[0].set_xlabel("x (cm)", fontsize=10)
    axs[0].set_ylabel("y (cm)", fontsize=10)
    axs[1].set_xlabel("y (cm)", fontsize=10)
    axs[1].set_ylabel("z (cm)", fontsize=10)
    for a_ in axs:
        a_.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-3d", action="store_true")
    opt = ap.parse_args()
    OUT = C.HERE / "results"
    OUT.mkdir(exist_ok=True)
    for kind in ("yagi", "horn"):
        fig3(OUT / f"geometry_fig3_{kind}.png", kind)
        if not opt.no_3d:
            fig_3d(OUT / f"geometry_3d_{kind}.png", kind)


if __name__ == "__main__":
    main()
