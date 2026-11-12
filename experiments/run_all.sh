#!/usr/bin/env bash
# Re-create every figure in figures/ and every number in results/.
# About 4 minutes on a laptop.  FAST=1 ./run_all.sh does a quicker pass
# with fewer Monte Carlo replications.
set -e
cd "$(dirname "$0")"
export MPLBACKEND=Agg PYTHONUTF8=1
for script in e1_taper_constants e2_theorem2_montecarlo e3_leakage e4_replication \
              e5_ext1_pooled_pitch e6_ext2_frame_length e7_ext3_strings; do
  echo "== $script"
  python3 "$script.py"
done
