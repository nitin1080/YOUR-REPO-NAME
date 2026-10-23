#!/usr/bin/env bash
# Sweep N and thread counts, write results/bench.csv and print a markdown table.
set -e
BIN=${1:-./build/matmul}
OUT=results/bench.csv; mkdir -p results
echo "N,threads,block,naive_s,blocked_s,threaded_s,false_sharing_s,max_err" > $OUT
for N in 256 512 1024 2048; do
  REPS=3; [ $N -ge 2048 ] && REPS=1
  for T in ${THREADS:-1 2 4 8}; do
    $BIN $N $T 64 --csv --reps=$REPS >> $OUT
  done
done
echo "| N | threads | naive (s) | blocked (s) | threaded (s) | false-sharing (s) | blocked ↑ | threaded ↑ |"
echo "|---|---|---|---|---|---|---|---|"
tail -n +2 $OUT | awk -F, '{printf "| %d | %d | %.3f | %.3f | %.3f | %.3f | %.2fx | %.2fx |\n",$1,$2,$4,$5,$6,$7,$4/$5,$4/$6}'
