# Yagi-Uda and horn phased arrays in MEEP

MEEP scripts that reproduce the results of

> J. C. Hanson, *Broadband RF Phased Array Design with MEEP: Comparisons to Array
> Theory in Two and Three Dimensions*, Electronics **2021**, 10, 415
> (`paper/electronics-10-00415-v2.pdf`).

| file          | contents                                                                         |
|---------------|----------------------------------------------------------------------------------|
| `yagi_uda.py` | Yagi-Uda element + arrays: 1D array in 2D (Sec. 3), 16 x 16 array in 3D (Sec. 4) |
| `horn.py`     | Exponential horn + arrays: 1D array in 2D (Sec. 3), 8 x 8 array in 3D (Sec. 4)   |
| `common.py`   | array theory (Eqs. 18, 25), near-to-far box, far-field cuts, beam metrics, plots |
| `plot_geometry.py` | geometry figures as in Fig. 3 (MEEP `plot2D`): `results/geometry_fig3_{yagi,horn}.{png,pdf}`, `geometry_3d_{yagi,horn}.{png,pdf}` |

These time-domain scripts are plain MEEP (FDTD time stepping). The same arrays solved in
the frequency domain with the MeepHoltz solvers (`solve_cw`, `solve_wh`) are described
under [Frequency-domain solvers](#frequency-domain-solvers-meepholtz).

## Layout

```
*.py        all scripts, side by side (time domain: common, horn, yagi_uda, plot_geometry;
            frequency domain: cw_common, horn_cw, yagi_cw, fast_bw, fig7_wh and the studies)
slurm/      Slurm jobs for TinkerCliff (bw2d, horn3d, submit_paper.sh) and local queues (run_*.sh)
results/    geometry figures; td/ time-domain runs and figures; cw/ frequency-domain runs,
            figures and studies; logs/
reports/    antennas/ (time domain), cw/ (frequency-domain solvers), fast_bw/ (3 px/cm sweep)
paper/      Hanson (2021) and the CPC paper outline
```

## Running

Run from this directory. On this machine MEEP is the MeepHoltz fork (install in `_local`):

```bash
export PYTHONPATH=~/Develop/MeepHoltz/_local/lib/python3.12/site-packages
PY=~/Develop/MeepHoltz/venv/bin/python
```

| paper result                                      | command                   | cost            |
|---------------------------------------------------|---------------------------|-----------------|
| Fig. 3: element and array geometries              | `$PY plot_geometry.py`    | ~1 min          |
| Fig. 4 top: Yagi steering, N = 8 and 16, 5 GHz    | `$PY yagi_uda.py fig4`    | 2D, ~1 min      |
| Fig. 5: Yagi N = 16 patterns, 2.5 / 5 GHz         | `$PY yagi_uda.py fig5`    | 2D, < 1 min     |
| Fig. 9: Yagi 16 x 16 steering grid, 5 GHz         | `$PY yagi_uda.py fig9`    | 3D, 36 runs     |
| Fig. 11: Yagi 16 x 16 E/H patterns, 3 / 4 GHz     | `$PY yagi_uda.py fig11`   | 3D, 6 runs      |
| Fig. 4 bottom: horn steering + beamwidth vs f     | `$PY horn.py fig4`        | 2D, few min     |
| Table 1 (right): horn scan loss at dPhi = 80 deg  | `$PY horn.py table1`      | 2D, ~1 min      |
| Fig. 6: horn N = 16 patterns, 0.5 / 5 GHz         | `$PY horn.py fig6`        | 2D, ~1 min      |
| Fig. 7: horn Ey snapshots 0.5-2 ns + pattern      | `$PY horn.py fig7`        | 2D, < 1 min     |
| Fig. 10: horn 8 x 8 steering + beamwidth vs f     | `$PY horn.py fig10`       | 3D, 32 runs     |
| Fig. 12: horn 8 x 8 E/H patterns, 0.5 / 1 GHz     | `$PY horn.py fig12`       | 3D, 6 runs      |

A single configuration:

```bash
$PY yagi_uda.py run --n 16 --freq 5 --dphi 40              # 2D, E-plane
$PY yagi_uda.py run --dim 3 --freq 4 --dphi 60 --dphi-h 30 # 3D, E- and H-plane
$PY horn.py run --freq 2.5 --dphi -35
$PY horn.py run --dim 3 --freq 1 --dphi 30 --dphi-h 60
```

`--help` on any subcommand lists all options (resolution, PML, run time, wall/wire
thickness, cell size, back-plane, ...). Everything runs under MPI unchanged
(`mpirun -np 8 $PY horn.py fig12`); only rank 0 writes.

**3D cost.** At the paper's resolutions the 3D cells are ~48 M voxels (Yagi 16 x 16,
res 6, 42 x 73 x 73 cm) and ~58 M voxels (horn 8 x 8, res 4, 40 x 150 x 150 cm); the
paper reports 11.7 GB for the horn case. Use MPI, or `--resolution` lower for a quick
look (`--resolution 2` for the horn and `4` for the Yagi still give the right beam angles).
`--no-subpixel` speeds up the set-up for the ~10^4 horn wall blocks considerably.

**Output.** Every run is cached as `results/td/runs/<case>_<hash>.npz` (angles, far-field
power per plane, all parameters in `meta`); the hash covers every parameter, so
changing an option re-runs and re-running a figure is free. `--force` ignores the
cache. Figures go to `results/td/*.png`, Table 1 to `results/td/horn_table1.txt`.

## Model

Units as in the paper (App. A): MEEP length unit = 1 cm, so `f_meep = f[GHz]/29.98`
and 6.67 ns = 200 time units. Metal is PEC (`mp.metal`), background is vacuum, PML
is 1 cm thick, sources are `ContinuousSource` switched on abruptly at t = 0, the
near-to-far box sits just inside the PML (the blue box of Fig. 3), and the far field
is evaluated at r = 1000 cm. Power is the radial Poynting flux; the E-plane is the xy-
plane (angle from +x toward +y), the H-plane the xz-plane (angle from +x toward +z,
i.e. the paper's H-plane shifted so broadside is 0 deg).

Element (i, j) (i along y, j along z, counted from the negative end) is driven with
amplitude `exp(-1j*(i*dPhiE + j*dPhiH))`. A positive phase shift then steers to a
negative angle, as in the paper's plots; the theory curve is Eq. (25) with
`sin(phi0) = -dPhi / (2 pi d / lambda)`.

**Yagi-Uda** (Table 1): reflector r = 3.75, driven element y = 2.81, three directors
z = 1.24, spacing d1 = 1.8, boom L = 7.2, dy = dz = 3.92, resolution 6. The driven
element is a y-directed line current (the red line of Fig. 3), the rest is metal.

**Horn** (Tables 1, 2): box of side a with back wall at x = 0, flare
`y = +-(a/2) exp(ln(2d/a)(x-a)/c)` from x = a to a + c built from n = c/dx straight
slices, radiator a y-directed line current across the box at x = a/2, dy = 2d.
2D: a = 0.95, c = 15, d = 3.8, dx = 0.1, res 6. 3D: a = 2, c = 15, d = 8, dx = 0.5,
dy = dz = 16, res 4, back-plane 142 x 142 x 0.5 at x = -2a.

### Choices the paper leaves open

These are defaults, all exposed as options:

- **Wire / wall thickness**: 0.2 cm (Yagi wires, 2D horn walls), 0.25 cm = 1 pixel
  (3D horn walls). The paper gives none.
- **3D horn has no top/bottom plates.** "The width in the z-direction ... remains
  fixed at a" is read as: every wall is extruded over height a in z. The horn is then
  a pair of flared parallel plates carrying a TEM mode. Closing it with plates at
  z = +-a/2 would make an a x a waveguide with a 7.5 GHz TE10 cutoff, which doesn't fit
  the 0.1-1.2 GHz band the paper reports.
- **Run time**: 6.67 ns (the paper's value for Fig. 7) but never fewer than 10 periods
  (`--min-periods`); at 0.3 GHz, 6.67 ns is only two periods.
- **Cell**: 42 cm (Yagi) / 40 cm (horn) along x, following Fig. 3; along y the array
  plus 4 cm (Yagi) / 6 cm (horn) on each side. Fig. 7 uses the paper's 80 x 150 cm cell.
- **Yagi back-plane**: off (Table 2 says the Yagi array needed no modification);
  `--backplane-x` adds one.
- **Beamwidth**: the -3 dB width of the simulated pattern. The paper's quoted
  11.0 / 5.0 deg for N = 8 / 16 correspond to 0.886 lambda/((N-1) d); Eq. (25) itself
  gives 0.886 lambda/(N d) (9.8 / 4.9 deg). `yagi_uda.py fig4` prints both.
- Real fields (a complex `amplitude` on a real-field run gives the phase-shifted
  cosine; checked to steer identically to `--complex`).

## Results obtained with the defaults (2D)

All 2D figures have been run (`results/td/`); the 3D ones (Figs. 9-12) have only been
smoke-tested at reduced resolution and array size (4 x 4, res 2-4: beam angles within
0.9 deg of theory in E and within 0.01 deg in H).

| quantity                                  | MEEP (here)     | Eq. (25) / Eq. (18) | paper          |
|-------------------------------------------|-----------------|---------------------|----------------|
| Yagi N = 16, 5 GHz: slope dphi/dPhi       | 0.2376          | 0.2434              | "slightly lower" |
| Yagi N = 8, 5 GHz: slope dphi/dPhi        | 0.2175          | 0.2434              | "slightly higher" |
| Yagi BW at dPhi = 0, N = 8 / 16           | 9.26 / 4.77 deg | 9.77 / 4.85 deg     | 11.0 / 5.0 deg |
| Horn BW = a/f + b (0.3-3 GHz)             | a = 11.5 ± 0.1, b = 0.6 ± 0.2 | -     | a = 12.0 ± 0.1, b = 1.1 ± 0.2 |
| Horn scan loss 0.5 / 1 / 2 / 4 GHz (dB)   | -10.4 / -0.9 / +0.1 / -0.9 | -        | -11.6 / -1.2 / -1.0 / -0.9 |
| Fig. 7: beam angle, BW (2.5 GHz, -35 deg) | 8.94, 5.22 deg  | 8.82, 5.07 deg      | 9, 5.5 ± 0.5 deg |

The N = 8 Yagi steers about 11% less than Eq. (18) predicts. Its wide beam is pulled
toward broadside by the element pattern. The paper reports a slightly *steeper* slope
there, which probably comes down to the unspecified wire thickness and driven-element
model. At 5 GHz the horn array has d/lambda = 1.27: Eq. (25) predicts full-height
grating lobes at about ±45 deg, and the horn element pattern suppresses them to about
-15 dB, as in the paper's Fig. 6.

Section 5 (dipoles in ice with graded n(z)) isn't implemented: it's neither a Yagi
nor a horn array.

## Frequency-domain solvers (MeepHoltz)

The same simulations solved with `solve_cw` (BiCGSTAB-L, complex fields) and `solve_wh`
(WaveHoltz, GMRES, real fields); see `cw_common.py` for how the far field is extracted.
Runs are cached in `results/cw/runs/`; the time-domain cache is read for comparison.

| result                                          | command                                   |
|-------------------------------------------------|-------------------------------------------|
| horn / Yagi figures, both solvers               | `$PY horn_cw.py bw` (or `fig4`, `table1`, `fig6`), `$PY yagi_cw.py fig4` |
| one case                                        | `$PY horn_cw.py run --freq 2.5 --dphi -35 --solvers wh --save-fields` |
| beamwidth sweep at 3 px/cm (~30 min, 6 cores)   | `$PY fast_bw.py run --record-sweep`, then `$PY fast_bw.py report` |
| paper Fig. 7 at t = infinity (solve_wh, 6 px/cm)| `$PY fig7_wh.py`                          |
| report `reports/cw/cw_report.pdf`               | `$PY make_report.py`                      |
| 3D horn on this machine when no MPI job runs    | `slurm/run_3d_idle.sh 1.0`                |
| all runs for the CPC paper, TinkerCliff         | `ACCOUNT=<alloc> slurm/submit_paper.sh`   |

Studies (each writes to `results/cw/<name>/` and the `reports/cw` tables):
`td_convergence.py`, `td_converged.py` (time-domain run length), `thickness_sensitivity.py`,
`wall_transparency.py`, `unknowns.py`, `radiator_compare.py`, `timing.py`.
