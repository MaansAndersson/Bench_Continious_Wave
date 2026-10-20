#!/bin/bash
# Submit the cavity benchmark on TinkerCliff, one job per node count.
#
#   ACCOUNT=<alloc> ./submit.sh strong      # nodes 1 2 4 8: P = 1..128 on one node, then 256..1024
#   ACCOUNT=<alloc> ./submit.sh weak
#   ACCOUNT=<alloc> ./submit.sh conv        # one node
#
# NODES overrides the node counts; every other cavity.slurm setting (RES, IFACES, SOLVERS, ...)
# is passed through from the environment, e.g.  RES=96 SOLVERS=wh ./submit.sh strong
set -euo pipefail
cd "$(dirname "$0")"

MODE=${1:?usage: submit.sh strong|weak|conv}
: "${ACCOUNT:?set ACCOUNT to your ARC allocation}"
case $MODE in
  strong|weak) NODES=${NODES:-1 2 4 8} ;;
  conv) NODES=${NODES:-1} ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

mkdir -p results
for n in $NODES; do
  sbatch --account="$ACCOUNT" --nodes="$n" --job-name="cavity-$MODE-n$n" \
    --export=ALL,MODE="$MODE" cavity.slurm
done
