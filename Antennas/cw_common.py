"""Frequency-domain solvers for the antenna arrays of horn.py and yagi_uda.py (Hanson 2021).

The geometry, sources, near-to-far box and beam analysis are those of the
time-domain scripts in the parent directory (imported, not copied); only the way
the time-harmonic field is obtained changes:

  td  time stepping from rest for until = max(6.67 ns, 10 periods), DFT from
      t = 0 (the parent scripts; their cached results are re-used, not re-run);
  cw  MEEP's frequency-domain solver fields::solve_cw (BiCGSTAB-L, L = 10),
      complex fields;
  wh  the MeepHoltz fork's WaveHoltz solver fields::solve_wh (GMRES(50) on
      I - S by default, or BiCGSTAB(L) with --wh-solver bicgstab; filter
      window = 2 periods), real fields.

Far field from a CW solution
----------------------------
cw: the near-to-far box is created *before* solve_cw.  MEEP suppresses DFT
    updates while solving (fields_chunk::doing_solve_cw) and calls update_dfts()
    once on exit, so the DFT holds exactly one sample of the converged complex
    phasor, which is what get_farfield needs (the overall scale is irrelevant:
    patterns are normalised and scan loss is a ratio within one solver).
    With the old meep-waveholtz build, time-stepping on from a solve_cw
    solution departed from exp(-i w t) u0 by 15-75 % (solve_cw returned the
    fields two steps late); MeepHoltz fixes this and `periodicity` for cw is
    ~1e-6 (2D Yagi, N = 4).  The n2f-before-solve scheme is kept regardless.
wh: solve_wh time-steps internally with DFT updates enabled, so a
    pre-existing DFT would be polluted.  Instead the near-to-far DFT is
    created after the solve, and exactly one window
    (periods * T = Nt whole steps, see `courant_for`) is stepped.  The WaveHoltz
    fixed point is the periodic solution of the discrete time stepping, so the
    DFT over the window returns its phasor exactly; `periodicity` (change of Ey
    over one window, max norm) is ~1e-6.

MeepHoltz notes (Simulation.solve_wh(tol, maxiters, L, periods, solver)):
* L is the GMRES restart length (--restart) or the BiCGSTAB(L) subspace
  dimension (--wh-L).
* solve_wh detaches the sources while applying I - S and restores them on
  exit, so they are still in place for the post-solve window.
* solve_wh rounds dt so that the window is Nt whole steps and restores the
  user dt on exit; choosing the Courant number up front makes both no-ops,
  so the post-solve stepping uses the dt the solution belongs to.
* It prints "Finished solve_wh after N GMRES|BiCGSTAB iters." and, from the
  Krylov solver, "final residual = r" (GMRES: true |b - Ax|/|b|; BiCGSTAB:
  the recursive residual), which are parsed into the run stats.
"""

import contextlib
import io
import json
import math
import pathlib
import re
import sys
import time

import numpy as np
import meep as mp

HERE = pathlib.Path(__file__).resolve().parent

import common as C  # noqa: E402

RESULTS = HERE / "results" / "cw"         # frequency-domain runs and figures
TD_RESULTS = HERE / "results" / "td"      # time-domain runs (horn.py, yagi_uda.py)
TD_RUNS = TD_RESULTS / "runs"
REPORTS = HERE / "reports"

SOLVERS = ("cw", "wh")
SOLVER_LABEL = {"td": "time domain", "cw": "solve_cw (BiCGSTAB-10)",
                "wh": "solve_wh (GMRES(50), 2 periods)"}


def wh_L(opts):
    """L argument of solve_wh: GMRES restart length or BiCGSTAB(L) dimension."""
    return opts.restart if opts.wh_solver == "gmres" else opts.wh_L


def wh_meta(opts):
    """Cache-key entries of a wh run (MeepHoltz solve_wh; old solve_waveholtz_cw runs miss)."""
    m = {"impl": "solve_wh", "periods": opts.periods, "farfield": "post-solve-window"}
    if opts.wh_solver == "gmres":
        m["restart"] = opts.restart
    else:
        m.update(wh_solver=opts.wh_solver, wh_L=opts.wh_L)
    return m


def courant_for(f_meep, resolution, periods, cmax=0.5, exact=False):
    """Largest Courant number <= cmax with periods/f an integer number of steps.

    exact=True adjusts the Courant number by a few ulps so that MEEP's dt = Courant/resolution
    equals solve_wh's T/Nt bit for bit, so that its set_dt is a no-op. (The old
    solve_waveholtz_cw called fields_chunk::changing_structure() on a 1-ulp difference, which
    segfaulted with conductivity; MeepHoltz rescales in place, so this is no longer required.)
    Used only for conductive-metal runs, so the cache keys of the PEC runs are unchanged.
    """
    T = periods * 1.0 / f_meep                    # as in cw_fields.cpp
    nt = math.ceil(T * resolution / cmax - 1e-9)
    c = T * resolution / nt
    if exact:
        target = T / nt
        for _ in range(64):
            dt = c / resolution
            if dt == target:
                break
            c = math.nextafter(c, math.inf if dt < target else -math.inf)
        assert c / resolution == T / nt, "could not match dt to T/Nt exactly"
    return c, nt


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def capture_stdout(echo=True):
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = _Tee(old, buf) if echo else buf
    try:
        yield buf
    finally:
        sys.stdout = old


def solve(sim, solver, fcen, opts):
    """Run solve_cw or solve_wh on an initialised sim; return stats."""
    sim.init_sim()
    sim._evaluate_dft_objects()
    t0 = time.time()
    with capture_stdout(echo=opts.verbose_solver) as log:
        if solver == "cw":
            ok = sim.fields.solve_cw(opts.tol, opts.maxiters_cw, opts.L)
        else:
            ok = sim.solve_wh(opts.tol, opts.maxiters_wh, wh_L(opts), opts.periods,
                              solver=mp.GMRES if opts.wh_solver == "gmres" else mp.BICGSTAB)
    wall = time.time() - t0
    txt = log.getvalue()
    if solver == "cw":
        m = re.findall(r"Finished solve_cw after (\d+) CG iters \(~ (\d+) timesteps\)", txt)
        iters, steps = (int(m[-1][0]), int(m[-1][1])) if m else (-1, -1)
    else:
        m = re.findall(r"Finished solve_wh after (\d+) (?:GMRES|BiCGSTAB) iters", txt)
        iters = int(m[-1]) if m else -1
        r = re.findall(r"final residual = ([0-9.eE+-]+|nan|inf)", txt)
        relres = float(r[-1]) if r else float("nan")
        nt = int(round(opts.periods / fcen / sim.fields.dt))
        # one operator application = one window of nt steps; + RHS (and the initial residual)
        if opts.wh_solver == "gmres":    # + one residual per restart cycle and the final one
            apps = iters + math.ceil(max(iters, 1) / opts.restart) + 2
        else:                            # 2L applications per BiCGSTAB(L) iteration
            apps = 2 * opts.wh_L * iters + 2
        steps = apps * nt if iters >= 0 else -1
    stats = {"converged": bool(ok), "iters": iters, "timesteps": steps, "wall_s": wall}
    if solver == "wh":
        stats["relres"] = relres
    if mp.am_master():
        print(f"   {solver}: converged={ok} iters={iters} ~{steps} timesteps {wall:.1f} s")
    return stats


def periodicity_error(sim, comp, window_steps, vol=None):
    """max|u(t+window) - u(t)| / max|u| after the solve (0 for an exact periodic solution),
    over the whole cell or over the plane vol (3D: whole volumes are too big to gather)."""
    kw = _vol_kw(vol)
    a = np.asarray(sim.get_array(component=comp, cmplx=not sim.fields.is_real, **kw))
    for _ in range(window_steps):
        sim.fields.step()
    b = np.asarray(sim.get_array(component=comp, cmplx=not sim.fields.is_real, **kw))
    return float(np.max(np.abs(b - a)) / max(np.max(np.abs(a)), 1e-300))


def field_planes(sim, dpml, n2f_gap):
    """{name: Volume} of the saved Ey planes; None means the whole (2D) cell.

    3D: the E-plane (xy) through the source row nearest z = 0, the H-plane (xz) through the
    source column nearest y = 0, and the yz-plane 0.5 cm inside the front (+x) face of the
    near-to-far box, i.e. the field the far-field transform sees.  Whole 3D volumes are not
    saved: get_array gathers them on every MPI rank (~1 GB per rank at 58 M cells).
    """
    cs = sim.cell_size
    if cs.z == 0:
        return {"xy": None}
    z_row = min({round(src.center.z, 9) for src in sim.sources}, key=abs)
    y_col = min({round(src.center.y, 9) for src in sim.sources}, key=abs)
    x_front = 0.5 * cs.x - dpml - n2f_gap - 0.5
    return {"xy": mp.Volume(center=mp.Vector3(0, 0, z_row), size=mp.Vector3(cs.x, cs.y, 0)),
            "xz": mp.Volume(center=mp.Vector3(0, y_col, 0), size=mp.Vector3(cs.x, 0, cs.z)),
            "yz": mp.Volume(center=mp.Vector3(x_front, 0, 0), size=mp.Vector3(0, cs.y, cs.z))}


def _vol_kw(vol):
    return {} if vol is None else {"center": vol.center, "size": vol.size}


def field_arrays(sim, planes, ey, ey_t0):
    """Saved solution on each plane: complex Ey, real Ey_t0, eps and pixel coordinates.

    ey and ey_t0 map plane name -> array.  2D runs (one plane, the whole cell) keep the keys
    Ey, Ey_t0, eps, x, y; 3D runs store Ey_<p>, Ey_t0_<p>, eps_<p>, u_<p>, v_<p> (the two
    in-plane coordinates) and pos_<p> (the plane's offset) for p in xy, xz, yz.
    The complex scale of Ey is arbitrary and differs between solvers (cw: the complex field at
    the end of the solve; wh: the DFT over the post-solve window); compare after a complex fit.
    Ey_t0 is the real, physical Ey right after the solve (same units as time stepping): for wh
    the periodic steady state at t = 0 mod the window, i.e. the t -> infinity field.
    """
    out = {}
    for name, vol in planes.items():
        x, y, z, _ = sim.get_array_metadata(**_vol_kw(vol))
        eps = np.squeeze(np.asarray(sim.get_array(component=mp.Dielectric, **_vol_kw(vol))))
        e, e0 = np.squeeze(np.asarray(ey[name], dtype=complex)), np.real(np.squeeze(ey_t0[name]))
        if vol is None:
            out.update(Ey=e, Ey_t0=e0, eps=eps, x=np.asarray(x), y=np.asarray(y))
            continue
        axes = {"xy": (x, y, vol.center.z), "xz": (x, z, vol.center.y), "yz": (y, z, vol.center.x)}
        u, v, pos = axes[name]
        out.update({f"Ey_{name}": e, f"Ey_t0_{name}": e0, f"eps_{name}": eps,
                    f"u_{name}": np.asarray(u), f"v_{name}": np.asarray(v),
                    f"pos_{name}": np.float64(pos)})
    return out


def cw_pattern(make_sim, solver, fcen, planes, opts, dpml, n2f_gap):
    """Build the sim (make_sim(courant, complex_fields)), solve, project to far field.

    With opts.save_fields the Ey solution is returned as well, on the planes of field_planes.
    """
    save = getattr(opts, "save_fields", False)
    courant, nt = courant_for(fcen, opts.resolution_used, opts.periods,
                              exact=getattr(opts, "exact_dt", False))
    sim = make_sim(courant, solver == "cw")
    box = C.near2far_box(sim.cell_size, dpml, n2f_gap)
    fplanes = field_planes(sim, dpml, n2f_gap)
    check = fplanes["xy"]                             # periodicity on the E-plane in 3D
    if solver == "cw":
        sim.init_sim()
        n2f = sim.add_near2far(fcen, 0, 1, *box)
        stats = solve(sim, solver, fcen, opts)        # evaluates n2f; one DFT sample on exit
        out = C.sim_farfield(sim, n2f, planes, opts.dang)
        if save:
            ey = {p: sim.get_array(component=mp.Ey, cmplx=True, **_vol_kw(v))
                  for p, v in fplanes.items()}
            out.update(field_arrays(sim, fplanes, ey, ey))
        stats["periodicity"] = periodicity_error(sim, mp.Ey, nt, check)   # diagnostic only
    else:
        stats = solve(sim, solver, fcen, opts)        # sources are restored by solve_wh
        ey_t0 = ({p: sim.get_array(component=mp.Ey, **_vol_kw(v)) for p, v in fplanes.items()}
                 if save else None)
        stats["periodicity"] = periodicity_error(sim, mp.Ey, nt, check)
        n2f = sim.add_near2far(fcen, 0, 1, *box)
        if save:                                      # phasor of Ey over the same window
            cs = sim.cell_size
            dft = {p: sim.add_dft_fields([mp.Ey], fcen, 0, 1,
                                         **(_vol_kw(v) or {"center": mp.Vector3(), "size": cs}))
                   for p, v in fplanes.items()}
        sim._evaluate_dft_objects()                   # DFT objects are lazy
        for _ in range(nt):
            sim.fields.step()                         # fields::step() also updates the DFTs
        out = C.sim_farfield(sim, n2f, planes, opts.dang)
        if save:
            ey = {p: sim.get_dft_array(d, mp.Ey, 0) for p, d in dft.items()}
            out.update(field_arrays(sim, fplanes, ey, ey_t0))
    out["stats_json"] = json.dumps(stats)
    sim.reset_meep()
    return out


# --------------------------------------------------------------------------
# Time-domain results of the parent scripts
# --------------------------------------------------------------------------

_TD_INDEX = None


def td_result(**match):
    """Cached time-domain run from results/td/runs whose meta matches `match`.

    Returns None if the parent scripts have not produced it.
    """
    global _TD_INDEX
    if _TD_INDEX is None:
        _TD_INDEX = []
        for p in sorted(TD_RUNS.glob("*.npz")):
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
            _TD_INDEX.append((p, meta))
    for p, meta in _TD_INDEX:
        if all(np.isclose(meta.get(k), v) if isinstance(v, (int, float)) and not isinstance(v, bool)
               else meta.get(k) == v for k, v in match.items()):
            with np.load(p, allow_pickle=False) as z:
                out = {k: z[k] for k in z.files}
            out["meta"] = meta
            return out
    return None


def stats_of(r):
    return json.loads(str(r["stats_json"])) if "stats_json" in r else {}


def solver_plan(a):
    """[(solver, only_cached)] in plot order: compute --solvers, show cached --show."""
    comp = [s for s in a.solvers.split(",") if s]
    show = [s for s in a.show.split(",") if s]
    return [(s, s not in comp) for s in SOLVERS if s in comp or s in show]


def cached_or(path, compute, a, only_cached):
    """C.cached, except that only_cached returns None instead of computing."""
    if only_cached and (a.force or not path.exists()):
        return None
    return C.cached(path, compute, a.force and not only_cached)


def add_solver_args(p):
    p.add_argument("--solvers", default="cw,wh",
                   help="comma list of solvers to compute: cw (solve_cw), wh (solve_wh);"
                        " '' computes nothing")
    p.add_argument("--show", default="cw,wh",
                   help="solvers to include in figures/tables when already cached")
    p.add_argument("--tol", type=float, default=1e-6, help="relative residual tolerance")
    p.add_argument("--L", type=int, default=10, help="BiCGSTAB-L parameter of solve_cw")
    p.add_argument("--maxiters-cw", type=int, default=20000)
    p.add_argument("--wh-solver", choices=["gmres", "bicgstab"], default="gmres",
                   help="Krylov method inside solve_wh")
    p.add_argument("--restart", type=int, default=50, help="GMRES restart of solve_wh")
    p.add_argument("--wh-L", type=int, default=2, help="BiCGSTAB(L) dimension of solve_wh")
    p.add_argument("--periods", type=int, default=2, help="WaveHoltz filter window in periods")
    p.add_argument("--maxiters-wh", type=int, default=2000,
                   help="max Krylov iterations of solve_wh")
    p.add_argument("--save-fields", action="store_true",
                   help="also store the Ey solution (complex, whole cell) in the cached run")
    p.add_argument("--verbose-solver", action="store_true",
                   help="echo the solvers' iteration output")
