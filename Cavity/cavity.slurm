#!/bin/bash
# Cavity benchmark on TinkerCliff: strong scaling, weak scaling, or grid convergence of
# solve_cw and solve_wh (MeepHoltz), through the C++ (./cavity) and Python (cavity.py) drivers.
#
# Submit through submit.sh (one job per node count), or directly, e.g.
#   sbatch --account=<alloc> --nodes=1 --export=ALL,MODE=strong cavity.slurm
#
# Every run writes results/<MODE>/<iface>_<solver>_np<P>_<extra>.log; `make analyze` collects the
# RESULT lines and the iteration counts from those logs.
#
# Settings (environment, defaults in brackets):
#   MODE        strong | weak | conv                                       [strong]
#   IFACES      interfaces to run                                          [cpp py]
#   SOLVERS     solvers to run                                             [wh cw]
#   WHSOLVER    Krylov method(s) inside solve_wh: gmres and/or bicgstab      [gmres]
#   PERIODS     solve_wh filter window(s) in source periods, e.g. "1 2 4"    [1]
#               (every WHSOLVER x PERIODS combination is run; cw runs once)
#   NPS         rank counts to run; default from the node count (below)
#   strong:     RES [64], SIZE ["2 2 2"]  -- fixed problem
#   weak:       RES [32], SIZE ["2 2 2"] is the box for P = 1; for P ranks it is scaled by
#               P = px*py*pz, the most cube-like factorization, so cells/rank stay exactly
#               fixed for any P (WEAK_KIND=box), or the resolution grows as P^(1/3) in a
#               fixed box (WEAK_KIND=res)                                      [box]
#   conv:       RES_LIST ["8 16 32 64 96"], SIZE ["2 2 2"]; ranks chosen per resolution
#   OMEGA       angular frequency/frequencies of the source, e.g. "1 1.5 2"  [1]
#               the whole sweep runs once per value; omega != 1 writes to results/<MODE>_w<omega>
#   TOL [1e-8]  EXTRA (more driver flags)  NOERROR (1: skip error)
#   LAUNCHER    srun | mpirun                                               [srun]
#   MPIRUN_FLAGS  ["--map-by core --bind-to core"]  (OpenMPI; fills nodes in order)
#   SRUN_FLAGS    ["--mpi=pmix --cpu-bind=cores"]  (PMIx is required on TinkerCliff)
#
#SBATCH --job-name=cavity
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0                 # all of the node's memory; srun steps otherwise get ~2 GB/core
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm-%x-%j.out

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1
# before `set -u`: module (Lmod) functions may reference unset variables
if ! source ./env_tinkercliff.sh; then
  echo "cavity.slurm: cannot source $PWD/env_tinkercliff.sh (submit from the Cavity directory)" >&2
  exit 1
fi
: "${MEEP_PREFIX:?env_tinkercliff.sh did not set MEEP_PREFIX}" "${MEEP_ROOT:?}" "${PYTHON:?}"
set -uo pipefail

MODE=${MODE:-strong}
IFACES=${IFACES:-cpp py}
SOLVERS=${SOLVERS:-wh cw}
WHSOLVER=${WHSOLVER:-gmres}
SIZE=${SIZE:-2 2 2}
OMEGA=${OMEGA:-1}
TOL=${TOL:-1e-8}
PERIODS=${PERIODS:-1}
EXTRA=${EXTRA:-}
LAUNCHER=${LAUNCHER:-srun}
MPIRUN_FLAGS=${MPIRUN_FLAGS:---map-by core --bind-to core}
SRUN_FLAGS=${SRUN_FLAGS:---mpi=pmix --cpu-bind=cores}
NNODES=${SLURM_JOB_NUM_NODES:-1}
CORES=${SLURM_NTASKS_PER_NODE:-128}
MAXNP=$((NNODES * CORES))
[ -x ./cavity ] || make cavity || exit 1

# Default rank counts: on one node 1, 2, 4, ... and CORES itself (also when CORES is not a power
# of two, e.g. 96-core nodes); on N > 1 nodes only the full N*CORES
# (the smaller counts are covered by the jobs with fewer nodes).
if [ -z "${NPS:-}" ]; then
  if [ "$NNODES" -eq 1 ]; then
    NPS=""; p=1
    while [ $p -lt "$CORES" ]; do NPS="$NPS $p"; p=$((p * 2)); done
    NPS="$NPS $CORES"
  else
    NPS=$MAXNP
  fi
fi

echo "== $MODE on $NNODES node(s), $(date), host $(hostname)"
echo "== MEEP_PREFIX=$MEEP_PREFIX  (MeepHoltz $(git -C "$MEEP_ROOT" rev-parse --short HEAD 2>/dev/null))"
module list 2>&1 | sed 's/^/   /'

# launch NP command...: start NP MPI ranks inside the allocation
launch() {
  local np=$1; shift
  case $LAUNCHER in
    mpirun) mpirun -n "$np" $MPIRUN_FLAGS "$@" ;;
    srun) srun $SRUN_FLAGS --ntasks="$np" --nodes="$(( (np + CORES - 1) / CORES ))" "$@" ;;
    *) echo "unknown LAUNCHER=$LAUNCHER (mpirun | srun)" >&2; return 1 ;;
  esac
}

# Solver variants: "cw", and "wh-<whsolver>-p<periods>" for every WHSOLVER x PERIODS pair.
VARIANTS=""
for s in $SOLVERS; do
  if [ "$s" = wh ]; then
    for w in $WHSOLVER; do for per in $PERIODS; do VARIANTS="$VARIANTS wh-$w-p$per"; done; done
  else
    VARIANTS="$VARIANTS $s"
  fi
done

# run TAG NP driver-args...: one run per iface/variant, logs in $OUT/<iface>_<variant>_TAG.log
run() {
  local tag=$1 np=$2; shift 2
  if [ "$np" -gt "$MAXNP" ]; then echo "skip $tag: needs $np > $MAXNP ranks"; return; fi
  for iface in $IFACES; do
    for v in $VARIANTS; do
      local args=(-solver "${v%%-*}" -omega "$W" -tol "$TOL")
      if [[ $v == wh-* ]]; then
        local w=${v#wh-}
        args+=(-whsolver "${w%-p*}" -periods "${v##*-p}")
      fi
      [ -n "${NOERROR:-}" ] && args+=(-noerror)
      local exe=(./cavity)
      [ "$iface" = py ] && exe=("$PYTHON" cavity.py)
      local l=$OUT/${iface}_${v}_$tag.log
      echo "-- $(date +%T) $iface $v np=$np $* -> $l"
      launch "$np" "${exe[@]}" "${args[@]}" "$@" $EXTRA > "$l" 2>&1
      local status=$?
      grep -h '^RESULT' "$l" || echo "   FAILED ($LAUNCHER exit $status), see $l"
    done
  done
}

read -r SX SY SZ <<< "$SIZE"

for W in $OMEGA; do
  # one results directory per omega: results/<MODE>, or results/<MODE>_w<omega> if omega != 1
  OUT=results/$MODE
  [ "$W" != 1 ] && OUT=${OUT}_w$W
  mkdir -p "$OUT"
  echo "== omega = $W -> $OUT"
  case $MODE in
    strong)
      RES=${RES:-64}
      for p in $NPS; do
        run "np${p}_r${RES}" "$p" -res "$RES" -sx "$SX" -sy "$SY" -sz "$SZ"
      done ;;

    weak)
      RES=${RES:-32}
      WEAK_KIND=${WEAK_KIND:-box}
      for p in $NPS; do
        if [ "$WEAK_KIND" = res ]; then
          r=$(awk -v r="$RES" -v p="$p" 'BEGIN { printf "%.4g", r * p^(1/3) }')
          run "np${p}_r${r}" "$p" -res "$r" -sx "$SX" -sy "$SY" -sz "$SZ"
        else
          # P = fx*fy*fz with fx >= fy >= fz as equal as possible (96 -> 6x4x4, 192 -> 8x6x4,
          # 768 -> 12x8x8; powers of two: 2 -> 2x1x1, 4 -> 2x2x1, 8 -> 2x2x2, 16 -> 4x2x2)
          read -r fx fy fz <<< "$(awk -v p="$p" 'BEGIN { best = p + 1
            for (c = 1; c * c * c <= p; c++) if (p % c == 0)
              for (b = c; b * b <= p / c; b++) if ((p / c) % b == 0) {
                a = p / c / b; if (a / c < best) { best = a / c; fx = a; fy = b; fz = c } }
            print fx, fy, fz }')"
          sx=$(awk -v a="$SX" -v f="$fx" 'BEGIN { print a * f }')
          sy=$(awk -v a="$SY" -v f="$fy" 'BEGIN { print a * f }')
          sz=$(awk -v a="$SZ" -v f="$fz" 'BEGIN { print a * f }')
          run "np${p}_r${RES}_${sx}x${sy}x${sz}" "$p" -res "$RES" -sx "$sx" -sy "$sy" -sz "$sz"
        fi
      done ;;

    conv)
      # ~32^3 cells per rank (power of two, capped by the allocation) keeps every run short
      for r in ${RES_LIST:-8 16 32 64 96}; do
        p=$(awk -v r="$r" -v sx="$SX" -v sy="$SY" -v sz="$SZ" -v m="$MAXNP" 'BEGIN {
              n = r * sx * r * sy * r * sz / 32768; p = 1; while (p * 2 <= n && p * 2 <= m) p *= 2; print p }')
        run "np${p}_r${r}" "$p" -res "$r" -sx "$SX" -sy "$SY" -sz "$SZ"
      done ;;

    *) echo "unknown MODE=$MODE (strong | weak | conv)"; exit 1 ;;
  esac
done

echo "== done $(date)"
