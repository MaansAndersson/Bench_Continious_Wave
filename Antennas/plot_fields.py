"""Plot the saved Ey solutions of cached frequency-domain runs (horn_cw.py / yagi_cw.py --save-fields).

For every saved plane: the real snapshot Ey(t) right after the solve (for solve_wh the t -> inf
steady state) and the magnitude |Ey| in dB, with metal in black.  2D runs have one plane (the
whole cell); 3D runs have the E-plane (xy), the H-plane (xz) and the yz-plane in front of the
array (see cw_common.field_planes).

    python plot_fields.py results/cw/fast_bw/runs/horn2d_wh_N16_f1.5_*.npz
    python plot_fields.py results/cw/paper/horn3d/runs            # every run with fields
    python plot_fields.py RUN.npz --out results/cw/fields --pdf
"""

import argparse
import json
import pathlib

import numpy as np
from matplotlib.colors import ListedColormap

import common as C

OUT = C.HERE / "results" / "cw" / "fields"
LABEL = {"xy": ("x (cm)", "y (cm)", "z"), "xz": ("x (cm)", "z (cm)", "y"),
         "yz": ("y (cm)", "z (cm)", "x")}


def load(path):
    with np.load(path, allow_pickle=False) as z:
        r = {k: z[k] for k in z.files}
    r["meta"] = json.loads(str(r["meta"]))
    return r


def planes(r):
    """[(name, Ey, Ey_t0 or None, eps, u, v, position or None)] of a run; [] without fields."""
    if "Ey" in r:
        return [("xy", r["Ey"], r.get("Ey_t0"), r["eps"], r["x"], r["y"], None)]
    return [(p, r[f"Ey_{p}"], r.get(f"Ey_t0_{p}"), r[f"eps_{p}"], r[f"u_{p}"], r[f"v_{p}"],
             float(r[f"pos_{p}"])) for p in ("xy", "xz", "yz") if f"Ey_{p}" in r]


def snapshot(ey, ey_t0):
    """Real field to show: the saved Ey(t) if present, else the phasor made real at its peak."""
    if ey_t0 is not None:
        return ey_t0
    return np.real(ey * np.exp(-1j * np.angle(ey.flat[np.argmax(np.abs(ey))])))


def title(m):
    s = "solve_wh" if m["solver"] == "wh" else "solve_cw"
    n = f"{m['n']} x {m['n']}" if m["dim"] == 3 else f"N = {m['n']}"
    return (f"{m['antenna']} {n}, {m['f_ghz']:g} GHz, $\\Delta\\Phi$ = ({m['dphi_e']:g}, "
            f"{m['dphi_h']:g}), {m['resolution']:g} px/cm, {s}")


def plot_run(r, fn, pdf=False):
    plt = C.pyplot()
    pl = planes(r)
    tall = np.ptp(pl[0][4]) < np.ptp(pl[0][5])             # 2D cells are taller than wide
    fig, axs = plt.subplots(len(pl), 2, squeeze=False,
                            figsize=(9, 6.5 * len(pl)) if tall else (11, 4.2 * len(pl)))
    for row, (name, ey, ey0, eps, u, v, pos) in zip(axs, pl):
        ext = [u[0], u[-1], v[0], v[-1]]
        metal = np.ma.masked_where(np.abs(eps - 1) < 1e-6, np.ones_like(eps, dtype=float))
        re = snapshot(ey, ey0)
        vmax = float(np.percentile(np.abs(re), 99.5)) or 1.0
        db = 20 * np.log10(np.maximum(np.abs(ey), 1e-300) / np.abs(ey).max())
        xl, yl, normal = LABEL[name]
        where = "" if pos is None else f" at {normal} = {pos:.1f} cm"
        for ax, (data, cmap, lo, hi, what) in zip(row, [
                (re, "RdBu_r", -vmax, vmax, "Re $E_y$" if ey0 is None else "$E_y(t)$"),
                (db, "inferno", -50.0, 0.0, "$|E_y|$ (dB)")]):
            im = ax.imshow(data.T, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi,
                           interpolation="nearest")
            ax.imshow(metal.T, origin="lower", extent=ext, cmap=ListedColormap(["k"]),
                      interpolation="nearest")
            ax.set_title(f"{what}, {name}-plane{where}", fontsize=11)
            ax.set_xlabel(xl, fontsize=10)
            ax.set_ylabel(yl, fontsize=10)
            ax.tick_params(labelsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title(r["meta"]), fontsize=11)
    fig.tight_layout()
    for f in [fn] + ([fn.with_suffix(".pdf")] if pdf else []):
        fig.savefig(f, dpi=150, bbox_inches="tight")
        print("wrote", f)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=pathlib.Path, help="run .npz files or folders of them")
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--pdf", action="store_true", help="also write PDF")
    a = ap.parse_args()
    files = [f for p in a.paths for f in (sorted(p.glob("*.npz")) if p.is_dir() else [p])]
    a.out.mkdir(parents=True, exist_ok=True)
    done = 0
    for f in files:
        r = load(f)
        if not planes(r):
            continue
        plot_run(r, a.out / (f.stem + ".png"), a.pdf)
        done += 1
    print(f"{done} of {len(files)} runs had saved fields")


if __name__ == "__main__":
    main()
