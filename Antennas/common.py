"""Shared pieces for the phased-array reproductions of Hanson, Electronics 2021, 10, 415.

  "Broadband RF Phased Array Design with MEEP: Comparisons to Array Theory in
   Two and Three Dimensions"  (paper/electronics-10-00415-v2.pdf)

Units (paper, Appendix A): the MEEP length unit is a = 1 cm and c = 1, so

    f_meep = f[GHz] / 29.98        t_meep = t[ns] * 29.98

Everything here is antenna-agnostic: array theory (Eqs. (18), (25)), the
near-to-far box, far-field cuts, beam angle / beamwidth / scan loss, caching of
run results, and the polar plots in the style of Figs. 5, 6, 11, 12.

Conventions
-----------
* The array lies along y (and z in 3D) and radiates toward +x (broadside).
* Element (i, j) is driven with complex amplitude exp(-1j*(i*dPhiE + j*dPhiH)),
  i, j = 0..N-1 counted from the -y / -z end.  A positive phase shift per
  element then steers the beam to a *negative* angle, as in the paper's
  Figs. 5, 6, 7 and 9 ("location of zero phase ... chosen to cause a negative
  beam angle").  Checked numerically: probe gives -9.5 deg vs -9.8 theory for
  N=8, d=3.92, f=5 GHz, dPhi=40 deg, with real and complex fields alike.
* E-plane = xy-plane, angle phi from +x toward +y.
  H-plane = xz-plane, angle psi from +x toward +z (the paper's H-plane
  "shifted so that the main beam occurs at 0", Sec. 2.3).
* Theory (Eq. 25) with sin(phi0) = -dPhi / (2 pi d / lambda).
"""

import hashlib
import json
import pathlib

import numpy as np
import meep as mp

C_CM_PER_NS = 29.9792458          # speed of light in cm/ns; MEEP a = 1 cm

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results" / "td"     # time-domain runs and figures


def metal(p):
    """PEC by default; a finite conductor if p["conductivity"] (MEEP D_conductivity) is set.

    The paper says all objects have "the same metallic conductivity" without a value. In MEEP
    units (a = 1 cm, c = 1) D_conductivity = 1 corresponds to about 0.27 S/m.
    """
    s = p.get("conductivity")
    return mp.metal if s is None else mp.Medium(epsilon=1, D_conductivity=s)


def ghz(f_ghz):
    """GHz -> MEEP frequency (a = 1 cm)."""
    return f_ghz / C_CM_PER_NS


def ns(t_ns):
    """ns -> MEEP time units (a = 1 cm)."""
    return t_ns * C_CM_PER_NS


def wavelength_cm(f_ghz):
    return C_CM_PER_NS / f_ghz


# --------------------------------------------------------------------------
# Array theory, Sec. 2
# --------------------------------------------------------------------------

def steer_angle(dphi_deg, d_over_lambda):
    """Exact beam angle (deg) for phase shift per element dphi_deg (Eq. 17 = 0).

    NaN when the requested steering is not a visible-space direction.
    """
    s = -np.deg2rad(dphi_deg) / (2 * np.pi * d_over_lambda)
    return np.rad2deg(np.arcsin(s)) if abs(s) <= 1 else np.nan


def linear_slope(d_over_lambda):
    """Small-angle slope d(dphi)/d(dPhi) of Eq. (18): lambda / (2 pi d)."""
    return 1.0 / (2 * np.pi * d_over_lambda)


def array_factor(angle_deg, n, d_over_lambda, dphi_deg):
    """Normalized array power pattern, Eq. (25), as a function of the cut angle."""
    s0 = -np.deg2rad(dphi_deg) / (2 * np.pi * d_over_lambda)
    x = np.pi * d_over_lambda * (np.sin(np.deg2rad(angle_deg)) - s0)
    num = np.sin(n * x)
    den = n * np.sin(x)
    small = np.abs(den) < 1e-12
    out = np.empty_like(x)
    out[~small] = (num[~small] / den[~small]) ** 2
    out[small] = 1.0
    return out


def element_phase(i, j, dphi_e_deg, dphi_h_deg=0.0):
    return np.exp(-1j * (i * np.deg2rad(dphi_e_deg) + j * np.deg2rad(dphi_h_deg)))


def array_positions(n, d):
    """Element centres along one axis, symmetric about 0."""
    return (np.arange(n) - (n - 1) / 2) * d


# --------------------------------------------------------------------------
# Near-to-far box and far-field cuts
# --------------------------------------------------------------------------

def near2far_box(cell, dpml, gap):
    """Closed Near2FarRegion box `gap` inside the PML (the blue box of Fig. 3)."""
    dims = 3 if cell.z > 0 else 2
    h = [0.5 * cell.x - dpml - gap, 0.5 * cell.y - dpml - gap,
         0.5 * cell.z - dpml - gap if dims == 3 else 0.0]
    regions = []
    for ax in range(dims):
        for sgn in (+1, -1):
            c = [0.0, 0.0, 0.0]
            c[ax] = sgn * h[ax]
            s = [2 * h[0], 2 * h[1], 2 * h[2]]
            s[ax] = 0.0
            regions.append(mp.Near2FarRegion(center=mp.Vector3(*c),
                                             size=mp.Vector3(*s), weight=sgn))
    return regions


def farfield_cut(sim, n2f, plane, angles_deg, r=1000.0):
    """Radial Poynting flux on a circle of radius r (paper: r = 1000 cm).

    plane = "E": (r cos phi, r sin phi, 0);  plane = "H": (r cos psi, 0, r sin psi).
    Must be called on every MPI rank (get_farfield is collective).
    """
    p = np.empty(len(angles_deg))
    for k, a in enumerate(np.deg2rad(angles_deg)):
        if plane == "E":
            u = np.array([np.cos(a), np.sin(a), 0.0])
        else:
            u = np.array([np.cos(a), 0.0, np.sin(a)])
        ff = sim.get_farfield(n2f, mp.Vector3(*(r * u)))
        e, h = np.array(ff[:3]), np.array(ff[3:])
        s = np.real(np.cross(e, np.conj(h)))
        p[k] = s @ u
    return p


def sim_farfield(sim, n2f, planes, dang, r=1000.0):
    angles = np.arange(-180.0, 180.0, dang)
    out = {"angles": angles}
    for pl in planes:
        out["P_" + pl] = farfield_cut(sim, n2f, pl, angles, r)
    return out


# --------------------------------------------------------------------------
# Beam metrics
# --------------------------------------------------------------------------

def beam_metrics(angles, power, window=(-90.0, 90.0), mask=None):
    """Peak angle (parabolic refinement), -3 dB beamwidth, peak power.

    The peak is searched in the forward half-space `window`; the -3 dB points
    are found by walking outwards from the peak (periodic in angle) and
    interpolating linearly in dB.
    """
    angles = np.asarray(angles)
    p = np.maximum(np.asarray(power), 1e-300)
    n = len(p)
    inwin = (angles >= window[0]) & (angles <= window[1])
    if mask is not None:
        inwin &= mask
    k = np.flatnonzero(inwin)[np.argmax(p[inwin])]
    db = 10 * np.log10(p / p[k])
    step = angles[1] - angles[0]
    y0, y1, y2 = db[(k - 1) % n], db[k], db[(k + 1) % n]
    den = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    peak = angles[k] + shift * step

    def edge(direction):
        for m in range(1, n // 2):
            j = (k + direction * m) % n
            if db[j] < -3.0:
                jp = (k + direction * (m - 1)) % n
                t = (-3.0 - db[jp]) / (db[j] - db[jp])
                return (m - 1 + t) * step
        return np.nan

    return {"peak_deg": float(peak), "bw_deg": float(edge(+1) + edge(-1)),
            "peak_power": float(p[k])}


def theory_metrics(n, d_over_lambda, dphi_deg, dang=0.01):
    """Metrics of the Eq. (25) main beam.

    For d > lambda/2 grating lobes are as tall as the main beam, so the peak is
    searched only within half a grating-lobe spacing (in sin-space) of the
    intended direction sin(phi0) = -dPhi / (2 pi d / lambda).
    """
    a = np.arange(-180.0, 180.0, dang)
    s0 = -np.deg2rad(dphi_deg) / (2 * np.pi * d_over_lambda)
    near = np.abs(np.sin(np.deg2rad(a)) - s0) < 0.5 / d_over_lambda
    return beam_metrics(a, array_factor(a, n, d_over_lambda, dphi_deg), mask=near)


def fit_inverse_f(f, bw):
    """Least-squares fit bw = a/f + b (Figs. 4 and 10); returns a, b and std errors."""
    A = np.column_stack([1.0 / np.asarray(f), np.ones(len(f))])
    coef, res, *_ = np.linalg.lstsq(A, bw, rcond=None)
    dof = max(len(f) - 2, 1)
    s2 = float(np.sum((A @ coef - bw) ** 2)) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return coef, np.sqrt(np.diag(cov))


# --------------------------------------------------------------------------
# Running and caching
# --------------------------------------------------------------------------

def cache_path(outdir, prefix, meta):
    """results/<prefix>_<hash of every parameter that affects the run>.npz"""
    key = hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:10]
    return pathlib.Path(outdir) / "runs" / f"{prefix}_{key}.npz"


def cached(path, compute, force=False):
    """Load results/<path>.npz, or compute (on every rank) and save on the master."""
    path = pathlib.Path(path)
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
        out["meta"] = json.loads(str(out["meta"]))
        if mp.am_master():
            print(f"[cache] {path.name}")
        return out
    out = compute()
    if mp.am_master():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: v for k, v in out.items() if k != "meta"},
                 meta=json.dumps(out["meta"]))
    return out


def run_pattern(sim, fcen, until, planes, dang, dpml, n2f_gap, farfield_r=1000.0):
    """Run to `until` with a near-to-far box and return the far-field cuts."""
    n2f = sim.add_near2far(fcen, 0, 1, *near2far_box(sim.cell_size, dpml, n2f_gap))
    sim.run(until=until)
    out = sim_farfield(sim, n2f, planes, dang, farfield_r)
    sim.reset_meep()
    return out


def default_until(f_ghz, until_ns, min_periods):
    """Paper runs 6.67 ns (Sec. 3.2); never fewer than `min_periods` periods."""
    return max(ns(until_ns), min_periods / ghz(f_ghz))


def add_common_args(p):
    p.add_argument("--resolution", type=float, default=None,
                   help="pixels per cm (default: paper value for the case)")
    p.add_argument("--dpml", type=float, default=1.0,
                   help="PML thickness in cm (paper: 1)")
    p.add_argument("--n2f-gap", type=float, default=0.5,
                   help="distance of the near-to-far box inside the PML (cm)")
    p.add_argument("--until-ns", type=float, default=6.67,
                   help="run time in ns (paper: 6.67 ns = 200 MEEP units)")
    p.add_argument("--min-periods", type=float, default=10,
                   help="run at least this many periods of the drive frequency")
    p.add_argument("--dang", type=float, default=0.25,
                   help="far-field angular sampling in degrees")
    p.add_argument("--complex", action="store_true",
                   help="complex fields (real fields suffice; 2x memory)")
    p.add_argument("--no-subpixel", action="store_true",
                   help="disable subpixel averaging (much faster set-up for horns)")
    p.add_argument("--force", action="store_true", help="ignore cached results")
    p.add_argument("--outdir", type=pathlib.Path, default=RESULTS)


# --------------------------------------------------------------------------
# Plotting (master rank only)
# --------------------------------------------------------------------------

def polar_pattern(ax, angles, power, theory, title, rmin=-40.0, peak=None):
    """Paper-style polar plot: MEEP in blue, Eq. (25) in red, both 0 dB at peak."""
    p = np.maximum(power, 1e-300)
    pk = beam_metrics(angles, p)["peak_power"] if peak is None else peak
    db = np.clip(10 * np.log10(p / pk), rmin, 0)
    th = np.clip(10 * np.log10(np.maximum(theory, 1e-300)), rmin, 0)
    a = np.deg2rad(np.append(angles, angles[0]))
    ax.plot(a, np.append(th, th[0]), "r", lw=1.0, label="Eq. (25)")
    ax.plot(a, np.append(db, db[0]), "b", lw=1.0, label="MEEP")
    ax.set_ylim(rmin, 0)
    ax.set_yticks([-30, -15, -3, 0])
    ax.set_yticklabels(["-30", "-15", "-3", "0"], fontsize=7)
    ax.set_thetagrids(range(0, 360, 45), fontsize=7)
    ax.set_title(title, fontsize=8)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def d_over_lambda(meta):
    return meta["dy"] / wavelength_cm(meta["f_ghz"])


def report(r, plane="E"):
    """Print and return (measured, theory) beam metrics for one cut of a run."""
    m = r["meta"]
    dphi = m["dphi_e"] if plane == "E" else m["dphi_h"]
    meas = beam_metrics(r["angles"], r["P_" + plane])
    th = theory_metrics(m["n"], d_over_lambda(m), dphi)
    print(f"   {plane}-plane: beam {meas['peak_deg']:+7.2f} deg (theory {th['peak_deg']:+7.2f}), "
          f"BW {meas['bw_deg']:6.2f} deg (theory {th['bw_deg']:6.2f})")
    return meas, th


def pattern_panel(ax, r, plane, title):
    m = r["meta"]
    dphi = m["dphi_e"] if plane == "E" else m["dphi_h"]
    th = array_factor(r["angles"], m["n"], d_over_lambda(m), dphi)
    polar_pattern(ax, r["angles"], r["P_" + plane], th, title)


def plot_run(r, fn):
    """Polar E (and H) patterns of a single run."""
    m = r["meta"]
    planes = "EH" if m["dim"] == 3 else "E"
    for pl in planes:
        report(r, pl)
    plt = pyplot()
    fig, axs = plt.subplots(1, len(planes), subplot_kw={"projection": "polar"},
                            figsize=(4 * len(planes), 4), squeeze=False)
    for ax, pl in zip(axs[0], planes):
        pattern_panel(ax, r, pl, f"{pl}-plane, f = {m['f_ghz']:g} GHz, "
                                 f"$\\Delta\\Phi$ = ({m['dphi_e']:g}, {m['dphi_h']:g})")
    axs[0, 0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print("wrote", fn)


def save(fig, fn):
    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    print("wrote", fn)
