"""3D PEC cavity with a manufactured CW solution, Python-interface twin of cavity.cpp.

Benchmarks solve_cw (BiCGSTAB(L), complex fields) and solve_wh (WaveHoltz, GMRES or BiCGSTAB(L)
on I - S, real fields) of the MeepHoltz fork. See cavity.cpp for the problem statement:
E = z^ u cos(w t), u = q(x)^3 q(y)^3 q(z)^2, q(s) = (s/a)^2 - 1, PEC walls at +-a.

    mpirun -n 4 python cavity.py -solver wh -res 16

Prints one line starting with "RESULT" on rank 0 (parsed by analyze.py).
"""
import argparse
import math
import time

import meep as mp

p = argparse.ArgumentParser()
p.add_argument("-solver", choices=["cw", "wh"], default="wh")
p.add_argument("-res", type=float, default=16)
p.add_argument("-sx", type=float, default=2)
p.add_argument("-sy", type=float, default=2)
p.add_argument("-sz", type=float, default=2)
p.add_argument("-omega", type=float, default=1.0, help="angular frequency [1]")
p.add_argument("-eps", type=float, default=1.0)
p.add_argument("-tol", type=float, default=1e-8)
p.add_argument("-maxiters", type=int, default=None)
p.add_argument("-L", type=int, default=None, help="BiCGSTAB L (cw) / GMRES restart (wh)")
p.add_argument("-periods", type=int, default=1, help="WaveHoltz filter window in periods")
p.add_argument("-whsolver", choices=["gmres", "bicgstab"], default="gmres",
               help="Krylov method inside solve_wh")
p.add_argument("-noerror", action="store_true", help="skip the error integral")
args = p.parse_args()

wh = args.solver == "wh"
maxiters = args.maxiters if args.maxiters is not None else (500 if wh else 10000)
L = args.L if args.L is not None else (50 if wh and args.whsolver == "gmres" else 2)  # GMRES restart 50, BiCGSTAB(2)
ax, ay, az = args.sx / 2, args.sy / 2, args.sz / 2
omega, eps = args.omega, args.eps
if omega <= 0:
    p.error("-omega must be positive")


def qpow(x, a, n):
    """q^n with q = (x/a)^2 - 1, and its first two x-derivatives."""
    s = x / a
    q = s * s - 1
    qn2 = q ** (n - 2)
    return q * q * qn2, 2 * n * s * q * qn2 / a, (2 * n * q + 4 * n * (n - 1) * s * s) * qn2 / a**2


def sol(r):
    X, Xx, Xxx = qpow(r.x, ax, 3)
    Y, Yy, Yyy = qpow(r.y, ay, 3)
    Z, Zz, _ = qpow(r.z, az, 2)
    return X * Y * Z, Xxx * Y * Z, X * Yyy * Z, Xx * Y * Zz, X * Yy * Zz


def jx(r):
    return -sol(r)[3] / omega


def jy(r):
    return -sol(r)[4] / omega


def jz(r):
    u, uxx, uyy, _, _ = sol(r)
    return omega * eps * u + (uxx + uyy) / omega


def now():
    mp.all_wait()
    return time.perf_counter()


# Courant <= 0.5 with an integer number of steps per period (see cavity.cpp).
T = 2 * math.pi / omega
Nt = math.ceil(2 * T * args.res - 1e-9)
courant = T * args.res / Nt
dt = courant / args.res
# drives J sin(w t): undoes the sinc(w dt/2) exp(-i w (t + dt/2)) of a continuous source
x = 0.5 * omega * dt
amp = 1j * complex(math.cos(x), math.sin(x)) * x / math.sin(x)

t0 = now()
src = mp.ContinuousSource(frequency=omega / (2 * math.pi))
cell = mp.Vector3(args.sx, args.sy, args.sz)
sim = mp.Simulation(
    cell_size=cell,
    resolution=args.res,
    Courant=courant,
    default_material=mp.Medium(epsilon=eps),
    force_complex_fields=not wh,
    sources=[mp.Source(src, c, center=mp.Vector3(), size=cell, amp_func=f, amplitude=amp)
             for c, f in [(mp.Ex, jx), (mp.Ey, jy), (mp.Ez, jz)]],
)
sim.init_sim()
t1 = now()

if wh:
    converged = sim.solve_wh(args.tol, maxiters, L, args.periods,
                             solver=mp.GMRES if args.whsolver == "gmres" else mp.BICGSTAB)
else:
    converged = sim.solve_cw(args.tol, maxiters, L)
t2 = now()

relerr = -1.0
if not args.noerror:
    err2 = sim.integrate_field_function(
        [mp.Ex, mp.Ey, mp.Ez],
        lambda r, ex, ey, ez: ex.real**2 + ey.real**2 + (ez.real - sol(r)[0]) ** 2)
    # same component list as err2 (and cavity.cpp), so both integrate on the same grid
    ref2 = sim.integrate_field_function([mp.Ex, mp.Ey, mp.Ez], lambda r, *_: sol(r)[0] ** 2)
    relerr = math.sqrt(err2.real / ref2.real)
t3 = now()

gv = sim.fields.gv
cells = gv.nx() * gv.ny() * gv.nz()
if mp.am_master():
    print(f"RESULT iface=py solver={args.solver} whsolver={args.whsolver if wh else '-'} np={mp.count_processors()} res={args.res:g} "
          f"sx={args.sx:g} sy={args.sy:g} sz={args.sz:g} omega={omega:.10g} eps={eps:g} "
          f"cells={cells} courant={courant:.6f} steps_per_period={Nt} tol={args.tol:g} L={L} "
          f"periods={args.periods if wh else 0} converged={int(converged)} "
          f"t_setup={t1 - t0:.4f} t_solve={t2 - t1:.4f} t_error={t3 - t2:.4f} "
          f"relerr={relerr:.6e}", flush=True)
