"""Unknowns of the paper that could explain the remaining differences.

Remaining differences (2D): the Yagi N = 8 steering slope (ours ~10 % below theory, paper
"slightly higher"), the horn beamwidth fit (a = 11.4 vs 12.0 deg GHz, b = 0.7 vs 1.1 deg) and
the scan loss at 1-2 GHz (ours ~ +0.25 dB, paper -1.2 / -1.0 dB).  Four candidates:

  runlength     the paper ran 6.67 ns at every frequency (App. A, Sec. 3.2); our TD runs use at
                least 10 periods.  TD with exactly 6.67 ns, all 2D configurations.
  pml           "a layer 1 dx unit thick": 1 cm (our reading) or 1 pixel; and how much a thick
                PML changes things.  WaveHoltz with PML 1/6 cm (1 pixel), 1 cm, 10 cm (interior
                cell unchanged).
  bwmethod      how the beamwidth was read off the far field: no re-runs, the cached patterns are
                resampled at 1 deg and the -3 dB edges taken at the nearest samples.
  conductivity  "the same metallic conductivity": metal as a finite conductor
                (D_conductivity 10, 100, 1e3, 1e4 in MEEP units; 1 ~ 0.27 S/m) instead of PEC.
                WaveHoltz.

Cases: horn N = 16 beamwidth sweep (0.3-3 GHz) and scan-loss pairs (0.5-4 GHz, dPhi 0/80);
Yagi N = 8 and 16 steering at 5 GHz (dPhi 0..80).  New runs are cached under
results/unknowns/ (WaveHoltz) and results/td/paper_runlength/ (TD).

    python unknowns.py runlength | pml | conductivity     # run one study
    python unknowns.py summary                            # tables + figure from the caches
"""

import argparse
import json

import numpy as np
import meep as mp

import cw_common as W
from cw_common import C
import horn as H
import horn_cw as HC
import yagi_cw as YC
import yagi_uda as Y
import timing as T

OUT_WH = W.RESULTS / "unknowns"
OUT_TD = W.TD_RESULTS / "paper_runlength"
GEN = W.REPORTS / "cw" / "generated"
DPS = np.arange(0, 81, 10.0)
SCAN_F = (0.5, 1.0, 2.0, 4.0)
PAPER = dict(slope8="> 0.2434", slope16="< 0.2434", a=12.0, b=1.1,
             sl={0.5: -11.6, 1.0: -1.2, 2.0: -1.0, 4.0: -0.9})
PMLS = (1 / 6, 1.0, 10.0)
SIGMAS = (10.0, 100.0, 1e3, 1e4)


def horn_cases():
    return sorted({(f, 0.0) for f in HC.BW_FREQS} | {(f, d) for f in SCAN_F for d in (0.0, 80.0)})


# --------------------------------------------------------------------------
# runs (each returns the result for one case; cached)
# --------------------------------------------------------------------------

def wh_horn(f, d, dpml=1.0, sigma=None, only_cached=False):
    default = dpml == 1.0 and sigma is None
    args = ["run", "--solvers", "" if only_cached else "wh", "--show", "wh" if only_cached else ""]
    if not default:
        args += ["--outdir", str(OUT_WH), "--dpml", f"{dpml:.10g}"]
    a, _ = HC.parse_args(args)
    if dpml != 1.0:
        a.sx = 38.0 + 2 * dpml           # keep the interior (x) of the default 40 cm cell
    if sigma is not None:
        a.conductivity = sigma
    return HC.simulate(a, "wh", f, d, only_cached=only_cached)


def wh_yagi(n, d, dpml=1.0, sigma=None, only_cached=False):
    default = dpml == 1.0 and sigma is None
    args = ["run", "--solvers", "" if only_cached else "wh", "--show", "wh" if only_cached else ""]
    if not default:
        args += ["--outdir", str(OUT_WH), "--dpml", f"{dpml:.10g}", "--sx", f"{40.0 + 2 * dpml:.10g}"]
    a, _ = YC.parse_args(args)
    if sigma is not None:
        a.conductivity = sigma
    return YC.simulate(a, "wh", n, 5.0, d, only_cached=only_cached)


def td(ant, n, f, d, paper_runlength):
    if not paper_runlength:
        return W.td_result(antenna=ant, dim=2, n=n, f_ghz=f, dphi_e=d, dphi_h=0.0,
                           resolution=6, dpml=1.0)
    a = T.td_ns(ant, OUT_TD, until_ns=6.67, min_periods=0)
    a.force = False
    return H.simulate(a, f, d) if ant == "horn" else Y.simulate(a, n, f, d, 0.0, 2)


def run_runlength():
    for ant, n, f, d in T.configs():
        td(ant, n, f, d, True)


def run_pml():
    for dpml in PMLS:
        if dpml == 1.0:
            continue
        for f, d in horn_cases():
            wh_horn(f, d, dpml=dpml)
        for n in (8, 16):
            for d in DPS:
                wh_yagi(n, d, dpml=dpml)


def run_conductivity():
    for s in SIGMAS:
        for n in (8, 16):
            for d in DPS:
                wh_yagi(n, d, sigma=s)
        for f, d in horn_cases():
            wh_horn(f, d, sigma=s)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def bw_coarse(r, step=1.0, plane="E"):
    """-3 dB width read off a pattern sampled every `step` degrees, nearest samples, no
    interpolation: the angular span of the samples within 3 dB of the peak, plus one step."""
    ang, p = r["angles"], r["P_" + plane]
    k = max(1, int(round(step / (ang[1] - ang[0]))))
    a, q = ang[::k], p[::k]
    fw = np.abs(a) <= 90
    i0 = np.flatnonzero(fw)[np.argmax(q[fw])]
    db = 10 * np.log10(q / q[i0])
    n = len(a)
    lo = hi = 0
    while db[(i0 - lo - 1) % n] >= -3:
        lo += 1
    while db[(i0 + hi + 1) % n] >= -3:
        hi += 1
    return (lo + hi + 1) * step


def quantities(get_horn, get_yagi, bw=None):
    """Slopes, beamwidth fit and scan losses for one variant; None where data are missing."""
    bw = bw or (lambda r: HC.metric(r)["bw_deg"])
    out = {}
    slope_th = C.linear_slope(Y.d_over_lambda(5.0))
    for n in (8, 16):
        rs = [get_yagi(n, d) for d in DPS]
        out[f"slope{n}"] = (float(np.polyfit(DPS, [abs(HC.metric(r)["peak_deg"]) for r in rs], 1)[0])
                            if all(r is not None for r in rs) else None)
        out[f"bw{n}"] = bw(rs[0]) if rs[0] is not None else None
    rs = [get_horn(f, 0.0) for f in HC.BW_FREQS]
    if all(r is not None for r in rs):
        (ka, kb), (sa, sb) = C.fit_inverse_f(HC.BW_FREQS, [bw(r) for r in rs])
        out.update(a=ka, sa=sa, b=kb, sb=sb)
    for f in SCAN_F:
        r0, r80 = get_horn(f, 0.0), get_horn(f, 80.0)
        out[f"sl{f}"] = (10 * np.log10(HC.metric(r80)["peak_power"] / HC.metric(r0)["peak_power"])
                         if r0 is not None and r80 is not None else None)
    out["slope_theory"] = slope_th
    return out


def variants():
    oc = dict(only_cached=True)
    V = [("WaveHoltz, PEC, PML 1 cm (reference)", lambda f, d: wh_horn(f, d, **oc),
          lambda n, d: wh_yagi(n, d, **oc), None),
         ("TD, max(6.67 ns, 10 periods) (first report)",
          lambda f, d: td("horn", 16, f, d, False), lambda n, d: td("yagi", n, 5.0, d, False), None),
         ("TD, exactly 6.67 ns (paper)", lambda f, d: _cached_td("horn", 16, f, d),
          lambda n, d: _cached_td("yagi", n, 5.0, d), None),
         ("WaveHoltz, BW at 1$^\\circ$ samples", lambda f, d: wh_horn(f, d, **oc),
          lambda n, d: wh_yagi(n, d, **oc), bw_coarse),
         ("TD 6.67 ns, BW at 1$^\\circ$ samples", lambda f, d: _cached_td("horn", 16, f, d),
          lambda n, d: _cached_td("yagi", n, 5.0, d), bw_coarse)]
    for dpml in PMLS:
        if dpml != 1.0:
            lab = "1 pixel" if dpml < 1 else f"{dpml:g} cm"
            V.append((f"WaveHoltz, PML {lab}",
                      lambda f, d, x=dpml: wh_horn(f, d, dpml=x, **oc),
                      lambda n, d, x=dpml: wh_yagi(n, d, dpml=x, **oc), None))
    for s in SIGMAS:
        V.append((f"WaveHoltz, conductor $\\sigma_D={s:g}$",
                  lambda f, d, x=s: wh_horn(f, d, sigma=x, **oc),
                  lambda n, d, x=s: wh_yagi(n, d, sigma=x, **oc), None))
    return V


def _cached_td(ant, n, f, d):
    a = T.td_ns(ant, OUT_TD, until_ns=6.67, min_periods=0)
    path = C.cache_path(a.outdir, "x", {})        # only to get the folder
    import glob
    for pth in glob.glob(str(path.parent / f"{ant}2d_N{n}_f{f:g}_E{d:g}_H0_*.npz")):
        with np.load(pth, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
        out["meta"] = json.loads(str(out["meta"]))
        return out
    return None


def fmt(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def summary():
    rows, data = [], {}
    for lab, gh, gy, bw in variants():
        q = quantities(gh, gy, bw)
        data[lab] = q
        rows.append(f"{lab} & {fmt(q['slope8'], 4)} & {fmt(q['slope16'], 4)} & "
                    f"{fmt(q.get('a'))} & {fmt(q.get('b'))} & " +
                    " & ".join(fmt(q[f'sl{f}']) for f in SCAN_F) + " \\\\")
    rows.append("\\midrule")
    rows.append("paper & $>0.2434$ & $<0.2434$ & 12.0 & 1.1 & " +
                " & ".join(f"{PAPER['sl'][f]:.1f}" for f in SCAN_F) + " \\\\")
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "tab_unknowns.tex").write_text("\n".join(rows) + "\n")
    (W.RESULTS / "unknowns_summary.json").write_text(json.dumps(data, indent=1))
    for lab, q in data.items():
        print(f"{lab:48s} slope8 {fmt(q['slope8'], 4)} slope16 {fmt(q['slope16'], 4)} "
              f"a {fmt(q.get('a'))} b {fmt(q.get('b'))} SL " +
              " ".join(fmt(q[f'sl{f}']) for f in SCAN_F))
    figure()


def figure():
    plt = C.pyplot()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    curves = [("WaveHoltz, PEC, PML 1 cm", lambda f, d: wh_horn(f, d, only_cached=True), "o-", "tab:blue"),
              ("TD, max(6.67 ns, 10 periods)", lambda f, d: td("horn", 16, f, d, False), "s--", "0.4"),
              ("TD, exactly 6.67 ns", lambda f, d: _cached_td("horn", 16, f, d), "x-", "k")]
    for dpml in PMLS:
        if dpml != 1.0:
            curves.append((f"WaveHoltz, PML {'1 pixel' if dpml < 1 else f'{dpml:g} cm'}",
                           lambda f, d, x=dpml: wh_horn(f, d, dpml=x, only_cached=True), "v-",
                           "tab:green" if dpml < 1 else "tab:olive"))
    for s, c in zip(SIGMAS, ("tab:red", "tab:orange", "tab:purple", "tab:brown")):
        curves.append((f"WaveHoltz, $\\sigma_D$ = {s:g}",
                       lambda f, d, x=s: wh_horn(f, d, sigma=x, only_cached=True), "^:", c))
    for lab, g, mk, col in curves:
        fs = [f for f in HC.BW_FREQS if g(f, 0.0) is not None]
        if fs:
            axs[0].plot(fs, [HC.metric(g(f, 0.0))["bw_deg"] for f in fs], mk, color=col, mfc="none",
                        ms=4, lw=0.8, label=lab)
        fs = [f for f in SCAN_F if g(f, 0.0) is not None and g(f, 80.0) is not None]
        if fs:
            axs[1].plot(fs, [10 * np.log10(HC.metric(g(f, 80.0))["peak_power"] /
                                           HC.metric(g(f, 0.0))["peak_power"]) for f in fs],
                        mk, color=col, mfc="none", ms=5, lw=0.8, label=lab)
    axs[1].plot(SCAN_F, [PAPER["sl"][f] for f in SCAN_F], "ks", ms=7, label="paper")
    axs[0].set_xlabel("Frequency (GHz)")
    axs[0].set_ylabel("Beam width (deg)")
    axs[0].set_title("Horn N = 16 beamwidth", fontsize=9)
    axs[1].set_xscale("log")
    axs[1].set_xticks(SCAN_F)
    axs[1].set_xticklabels([f"{f:g}" for f in SCAN_F])
    axs[1].xaxis.set_minor_formatter(plt.NullFormatter())
    axs[1].set_xlabel("Frequency (GHz)")
    axs[1].set_ylabel(r"scan loss at $\Delta\Phi$ = 80 (dB)")
    axs[1].set_title("Horn scan loss", fontsize=9)
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle("Unknowns of the paper: run length, PML thickness, metal conductivity", fontsize=10)
    fig.tight_layout()
    fn = W.RESULTS / "unknowns.png"
    fig.savefig(fn, dpi=150)
    print("wrote", fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["runlength", "pml", "conductivity", "summary"])
    cmd = ap.parse_args().cmd
    OUT_WH.mkdir(parents=True, exist_ok=True)
    OUT_TD.mkdir(parents=True, exist_ok=True)
    {"runlength": run_runlength, "pml": run_pml, "conductivity": run_conductivity,
     "summary": summary}[cmd]()


if __name__ == "__main__":
    main()
