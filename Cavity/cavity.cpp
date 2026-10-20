/* 3D PEC cavity with a manufactured CW solution: scaling/convergence benchmark for
   fields::solve_cw (BiCGSTAB(L), complex fields) and fields::solve_wh (WaveHoltz, GMRES or
   BiCGSTAB(L) on I - S, real fields) of the MeepHoltz fork.  Same problem as cavity.py.

   Cell [-sx/2, sx/2] x [-sy/2, sy/2] x [-sz/2, sz/2], PEC walls (MEEP's default boundary),
   uniform eps.  Exact solution E = z^ u cos(w t), with s = x/(sx/2) etc. and q(s) = s^2 - 1:

     u = q(x)^3 q(y)^3 q(z)^2

   forced by J sin(w t), J = w eps E - curl curl E / w = (-u_xz, -u_yz, w^2 eps u + u_xx + u_yy) / w.
   The exponents give every normal current a double zero at its wall, which sidesteps the 7/8
   boundary weight add_volume_source applies half a cell inside the wall (meep-waveholtz
   DEAA/STATUS.md).

   usage: cavity [options]           (all ranks get the same arguments)
     -solver cw|wh   -res R   -sx/-sy/-sz S   -omega W (angular frequency) [1]   -eps E
     -tol T   -maxiters N   -L L (BiCGSTAB L [2] / GMRES restart [50])   -periods P (WaveHoltz window)
     -whsolver gmres|bicgstab (Krylov method inside solve_wh)
     -noerror (skip the error integral)
   One line starting with "RESULT" is printed on rank 0; analyze.py parses it. */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <meep.hpp>

using namespace meep;
using std::complex;

static const double PI = 3.14159265358979323846;

static double g_ax, g_ay, g_az; // half widths
static double g_omega, g_eps;

/* f = q^p with q = (x/a)^2 - 1, and its first two x-derivatives. */
static void qpow(double x, double a, int p, double &f, double &df, double &d2f) {
  const double s = x / a, q = s * s - 1;
  const double qp2 = std::pow(q, p - 2);
  f = q * q * qp2;
  df = 2 * p * s * q * qp2 / a;
  d2f = (2 * p * q + 4 * p * (p - 1) * s * s) * qp2 / (a * a);
}

struct sol_vals {
  double u, uxx, uyy, uxz, uyz;
};
static sol_vals sol(const vec &p) {
  double X, Xx, Xxx, Y, Yy, Yyy, Z, Zz, Zzz;
  qpow(p.x(), g_ax, 3, X, Xx, Xxx);
  qpow(p.y(), g_ay, 3, Y, Yy, Yyy);
  qpow(p.z(), g_az, 2, Z, Zz, Zzz);
  return {X * Y * Z, Xxx * Y * Z, X * Yyy * Z, Xx * Y * Zz, X * Yy * Zz};
}

static complex<double> Jx(const vec &p) { return -sol(p).uxz / g_omega; }
static complex<double> Jy(const vec &p) { return -sol(p).uyz / g_omega; }
static complex<double> Jz(const vec &p) {
  const sol_vals s = sol(p);
  return g_omega * g_eps * s.u + (s.uxx + s.uyy) / g_omega;
}
static double eps_fn(const vec &) { return g_eps; }

/* |E - E_exact|^2 and |E_exact|^2 on the integration grid; for complex (solve_cw) fields the
   physical field is the real part. */
static complex<double> err2(const complex<realnum> *F, const vec &p, void *) {
  const double ex = real(F[0]), ey = real(F[1]), ez = real(F[2]) - sol(p).u;
  return ex * ex + ey * ey + ez * ez;
}
static complex<double> ref2(const complex<realnum> *, const vec &p, void *) {
  const double u = sol(p).u;
  return u * u;
}

static double now() {
  all_wait();
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 1;

  std::string solver = "wh", whsolver = "gmres";
  double res = 16, sx = 2, sy = 2, sz = 2, tol = 1e-8;
  int maxiters = -1, L = -1, periods = 1;
  bool do_error = true;
  g_omega = 1.0;
  g_eps = 1.0;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) meep::abort("missing value for %s", a.c_str());
      return argv[++i];
    };
    if (a == "-solver") solver = next();
    else if (a == "-res") res = atof(next());
    else if (a == "-sx") sx = atof(next());
    else if (a == "-sy") sy = atof(next());
    else if (a == "-sz") sz = atof(next());
    else if (a == "-omega") g_omega = atof(next());
    else if (a == "-eps") g_eps = atof(next());
    else if (a == "-tol") tol = atof(next());
    else if (a == "-maxiters") maxiters = atoi(next());
    else if (a == "-L") L = atoi(next());
    else if (a == "-periods") periods = atoi(next());
    else if (a == "-whsolver") whsolver = next();
    else if (a == "-noerror") do_error = false;
    else meep::abort("unknown argument %s", a.c_str());
  }
  if (solver != "cw" && solver != "wh") meep::abort("-solver must be cw or wh");
  if (!(g_omega > 0)) meep::abort("-omega must be positive");
  if (whsolver != "gmres" && whsolver != "bicgstab")
    meep::abort("-whsolver must be gmres or bicgstab");
  const bool wh = solver == "wh";
  if (maxiters < 0) maxiters = wh ? 500 : 10000;
  if (L < 0) L = wh && whsolver == "gmres" ? 50 : 2; // GMRES restart 50, BiCGSTAB(2)
  g_ax = sx / 2, g_ay = sy / 2, g_az = sz / 2;

  /* Courant <= 0.5 chosen so that one source period is an integer number of steps: the
     WaveHoltz filter then needs no dt adjustment, and both solvers use the same dt. */
  const double T = 2 * PI / g_omega;
  const int Nt = (int)std::ceil(2 * T * res - 1e-9);
  const double courant = T * res / Nt;
  const double dt = courant / res;

  const double t0 = now();
  grid_volume gv = vol3d(sx, sy, sz, res);
  gv.center_origin();
  structure s(gv, eps_fn, no_pml(), identity(), 0, courant);
  fields f(&s);
  if (wh) f.use_real_fields();

  /* A continuous source drives sinc(w dt/2) exp(-i w (t + dt/2)) J; this amplitude turns it
     into J sin(w t) (as add_cw_source in DEAA/cpp/wh.cpp). */
  continuous_src_time src(g_omega / (2 * PI), 0.0, 0.0);
  src.is_integrated = false;
  const double x = 0.5 * g_omega * dt;
  const complex<double> amp = complex<double>(0, 1) * std::exp(complex<double>(0, x)) * x / std::sin(x);
  f.add_volume_source(Ex, src, f.v, Jx, amp);
  f.add_volume_source(Ey, src, f.v, Jy, amp);
  f.add_volume_source(Ez, src, f.v, Jz, amp);
  const double t1 = now();

  const bool converged =
      wh ? f.solve_wh(tol, maxiters, L, periods, whsolver == "gmres" ? GMRES : BICGSTAB)
         : f.solve_cw(tol, maxiters, L);
  const double t2 = now();

  double relerr = -1;
  if (do_error) {
    const component cs[3] = {Ex, Ey, Ez};
    const double e2 = real(f.integrate(3, cs, err2, NULL, f.total_volume()));
    const double r2 = real(f.integrate(3, cs, ref2, NULL, f.total_volume()));
    relerr = std::sqrt(e2 / r2);
  }
  const double t3 = now();

  master_printf("RESULT iface=cpp solver=%s whsolver=%s np=%d res=%g sx=%g sy=%g sz=%g omega=%.10g eps=%g "
                "cells=%.0f courant=%.6f steps_per_period=%d tol=%g L=%d periods=%d "
                "converged=%d t_setup=%.4f t_solve=%.4f t_error=%.4f relerr=%.6e\n",
                solver.c_str(), wh ? whsolver.c_str() : "-", count_processors(), res, sx, sy, sz, g_omega, g_eps,
                (double)gv.nx() * gv.ny() * gv.nz(), courant, Nt, tol, L, wh ? periods : 0,
                (int)converged, t1 - t0, t2 - t1, t3 - t2, relerr);
  return 0;
}
