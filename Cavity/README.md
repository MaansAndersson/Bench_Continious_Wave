In this folder we have to testcases of a continious wave sourced 3D cavity problem with an aanalytical solution implemented in MEEP using the C++ interface and the Python interface.
The main task is to evaluate the scalability of solve_cw and solve_wh (a solver in a specific fork of MEEP).
We should also look at the convergence of the solvers.

Create a suitable slurm script to be used on tinkercliff at ARC.
We need both the strong scaling and the weak scaling of the problem.




## Implementation

Both drivers solve the same problem and print the same `RESULT` line.

* `cavity.cpp` uses the C++ interface; `cavity.py` uses the Python interface.
* The solvers come from the **MeepHoltz** fork:
  * `solve_cw`: BiCGSTAB(L) with complex fields.
  * `solve_wh`: WaveHoltz with real fields, solved by GMRES or BiCGSTAB (`-whsolver`).

**Problem.** A PEC box `[-sx/2, sx/2] x [-sy/2, sy/2] x [-sz/2, sz/2]` with uniform eps. The exact solution is
`E = z^ u cos(wt)`, where `u = q(x)^3 q(y)^3 q(z)^2` and `q(s) = (2s/size)^2 - 1`.
It is forced by the current `J sin(wt)`, with `J = (w eps E - curl curl E / w)`.
The exponents give each normal current a double zero at its wall. This avoids MEEP's 7/8
boundary weighting of volume sources. The error is the relative L2 norm of `E - E_exact` over the cell.

Driver flags: `-solver cw|wh  -res R  -sx/-sy/-sz  -omega  -eps  -tol  -maxiters  -L  -periods  -whsolver gmres|bicgstab  -noerror`.

### Files

| file | purpose |
|---|---|
| `cavity.cpp`, `cavity.py` | the benchmark (C++ / Python interface) |
| `Makefile` | build, local smoke test (`make check`) and local convergence run (`make convergence`), submission (`make strong|weak|conv`), analysis (`make analyze`) |
| `env_tinkercliff.sh` | modules and paths on TinkerCliff (check module names with `module spider`) |
| `cavity.slurm` | one job: strong / weak / conv sweep over rank counts, all ifaces x solvers |
| `submit.sh` | submits one `cavity.slurm` job per node count (1 2 4 8 by default) |
| `analyze.py` | logs -> `summary.csv`, speedup/efficiency/order tables, plots |

### On TinkerCliff

1. Build MeepHoltz with MPI and Python into `$MEEP_ROOT/_local`, using the modules in `env_tinkercliff.sh`.
2. Build and submit:

   ```bash
   source env_tinkercliff.sh && make
   ACCOUNT=<alloc> ./submit.sh strong   # res 64, 2x2x2 box; P = 1..128 on 1 node, then 256, 512, 1024
   ACCOUNT=<alloc> ./submit.sh weak     # 32^3 cells/rank at res 32; box doubled in x, y, z in turn
   ACCOUNT=<alloc> ./submit.sh conv     # res 8..96
   make analyze
   ```

   Options are passed as environment variables, e.g. `RES=96 SOLVERS=wh IFACES=cpp ./submit.sh strong`.
   See the header of `cavity.slurm` for all of them.
   * **`solve_wh` method and window:** `WHSOLVER` (`gmres` and/or `bicgstab`) and `PERIODS`
     (filter window in source periods) take lists, and every combination is run. For example:

     ```bash
     WHSOLVER="gmres bicgstab" PERIODS="1 2 4" ACCOUNT=<alloc> ./submit.sh strong
     make convergence WHSOLVER="gmres bicgstab" PERIODS="1 2"   # locally
     ```

     `solve_cw` runs once per configuration. Each combination gets its own log
     (`<iface>_wh-<method>-p<periods>_...log`) and its own plot series.
   * **Krylov parameter `-L`:** the drivers default to GMRES restart 50 and BiCGSTAB(2).
     Other values go through `EXTRA="-L 4"`; note that `EXTRA` also applies to `solve_cw`.
   * `WEAK_KIND=res` keeps the box fixed and grows the resolution as `P^(1/3)`.
     Note that the steps per period then grow with the resolution.

**Interpreting weak scaling.** In `box` mode the cavity grows electrically: at 1024 ranks it is
32x16x16 at w = 1, so modes come close to the source frequency and the iteration counts can grow.
`analyze.py` therefore reports both the efficiency of the total solve and the time per iteration.
The time per FDTD step is the cleaner measure of parallel efficiency.

**Comparing iteration counts.** Counts are not comparable between Krylov methods. One
BiCGSTAB(L) iteration applies the operator 2L times; one GMRES iteration applies it once.
`analyze.py` therefore also reports:
* `applies`: the number of operator applications.
* `steps`: the FDTD time steps those applications cost. One WaveHoltz application is
  `periods x steps_per_period` steps; one `solve_cw` application is one step.

### Local results (MeepHoltz d24b62a4, 4 ranks, `make convergence NP=4`)

| res | rel. L2 error (all four) | order | solve_wh GMRES its | solve_cw BiCGSTAB its |
|---|---|---|---|---|
| 8 | 2.18e-2 | | 8 | 72 |
| 16 | 6.45e-3 | 1.78 | 8 | 162 |
| 24 | 3.07e-3 | 1.83 | 8 | 239-268 |
| 32 | 1.80e-3 | 1.87 | 8 | 325 |

`solve_wh` with other settings (C++, 4 ranks, operator applications at res 8 / 16 / 24):

| method | window | applications |
|---|---|---|
| GMRES | 1 period | 8 / 8 / 8 |
| GMRES | 2 periods | 7 / 7 / 7 |
| BiCGSTAB(2) | 1 period | 12 / 12 / 12 |
| BiCGSTAB(2) | 2 periods | 12 / 8 / 8 |

All settings reach the same error.

C++ and Python give identical errors, and so do both solvers.

* The WaveHoltz iteration count does not depend on the resolution.
* The `solve_cw` count grows roughly linearly with the resolution.
* At res 32, `solve_wh` takes 4.9 s against 23 s for `solve_cw`.
