"""Is a thin PEC sheet opaque on the grid?  Transmission vs thickness and sub-pixel offset.

An infinite PEC wall (mp.metal, subpixel averaging on, as in all antenna runs) of thickness
t separates a plane-wave source from a flux plane; 2D, periodic along the wall, 3 GHz.
For each t the wall is shifted through 8 sub-pixel offsets, because the horn flare and the
Yagi wires sit at arbitrary positions relative to the grid.  Transmission is normalised to
the run without wall.

Result at the paper's resolution (6 pixels/cm): a sheet thinner than one pixel (0.167 cm)
is transparent (T = 1) at some offsets, so thinner walls or wires are not represented
reliably.

    python wall_transparency.py        # results/wall_transparency.json + report table
"""

import json

import meep as mp

import cw_common as W

F = 3.0 / 29.9792458
THICK = (0.1, 0.15, 1 / 6, 0.2, 0.25, 0.3, 0.4)
OFFSETS = 8


def flux(t, res, offset=0.0):
    geom = [] if t is None else [mp.Block(center=mp.Vector3(offset), size=mp.Vector3(t, mp.inf),
                                          material=mp.metal)]
    sim = mp.Simulation(cell_size=mp.Vector3(30, 10), resolution=res, geometry=geom,
                        boundary_layers=[mp.PML(3, direction=mp.X)], k_point=mp.Vector3(),
                        sources=[mp.Source(mp.GaussianSource(F, fwidth=0.5 * F), component=mp.Ey,
                                           center=mp.Vector3(-8), size=mp.Vector3(0, 10))])
    fl = sim.add_flux(F, 0, 1, mp.FluxRegion(center=mp.Vector3(8), size=mp.Vector3(0, 10)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ey, mp.Vector3(8), 1e-6))
    out = mp.get_fluxes(fl)[0]
    sim.reset_meep()
    return out


def main():
    rows = []
    for res in (6, 12):
        ref = flux(None, res)
        for t in THICK:
            T = [flux(t, res, k / res / OFFSETS) / ref for k in range(OFFSETS)]
            rows.append(dict(res=res, t=t, pixels=t * res, T_max=max(T),
                             transparent=sum(x > 1e-3 for x in T)))
            if mp.am_master():
                print(f"res {res:2d}  t = {t:.3f} cm ({t * res:.2f} px): max T {max(T):.1e}, "
                      f"transparent at {rows[-1]['transparent']}/{OFFSETS} offsets")
    if mp.am_master():
        (W.RESULTS / "wall_transparency.json").write_text(json.dumps(rows, indent=1))
        gen = W.REPORTS / "cw" / "generated"
        gen.mkdir(parents=True, exist_ok=True)
        lines = []
        for t in THICK:
            r6 = next(r for r in rows if r["res"] == 6 and r["t"] == t)
            r12 = next(r for r in rows if r["res"] == 12 and r["t"] == t)
            lines.append(f"{t:.3f} & {r6['pixels']:.2f} & {r6['transparent']}/{OFFSETS} & "
                         f"{r12['pixels']:.2f} & {r12['transparent']}/{OFFSETS} \\\\")
        (gen / "tab_wall_transparency.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
