# High-Performance Matrix Multiplication (C++17)

Naive, cache-blocked and multithreaded N×N `double` matrix multiplication, written from scratch (no external libraries). Includes a cache simulator for memory-access analysis, a deliberate false-sharing kernel, and a reproducible benchmark script.

## Structure
```
matmul.hpp          kernels: naive, blocked, threaded, threaded_false_sharing
main.cpp            benchmark + correctness check vs naive
cache_sim.cpp       LRU set-associative cache model: replays each loop nest's address stream
false_sharing.cpp   packed vs alignas(64) counters micro-benchmark
bench.sh            sweeps N and thread count -> results/bench.csv + markdown table
CMakeLists.txt      Release (-O2 -g), optional -DENABLE_GPROF=ON
```

## Build & Run
```bash
cmake -B build && cmake --build build
./build/matmul [N=1024] [threads=hw] [block=64]     # all kernels, GFLOP/s, speedups
./build/cache_sim [N=256] [BS=32] [cacheKB=32]      # miss-rate analysis
./build/false_sharing
./bench.sh build/matmul                              # full sweep
```

## Kernels
| Kernel | Idea |
|---|---|
| `mul_naive` | ijk; `B[k][j]` walks a column (stride N) → almost every access hits a new cache line |
| `mul_blocked` | BS×BS tiling + ikj order; all inner accesses unit-stride, a tile of B is reused from L1/L2 |
| `mul_threaded` | blocked kernel; each `std::thread` owns a disjoint band of C rows (multiple of BS) → no cache line written by two threads |
| `mul_threaded_false_sharing` | intentionally bad: threads interleave columns, so neighbours write adjacent `C[i][j]` in the same 64B line |

## Memory-Access Analysis
`cache_sim` replays the exact address stream through a 32 KB, 8-way, 64 B-line LRU cache:

| N | Naive miss rate | Blocked miss rate | Misses (naive → blocked) |
|---|---|---|---|
| 256 | 50.3% | 4.4% | 16.9M → 2.2M (7.6× fewer) |
| 512 | 50.2% | 4.4% | 134.8M → 17.8M (7.6× fewer) |

Naive misses on every `B` access (half of all accesses); blocking keeps the miss rate flat as N grows.

## Benchmarks
Intel Xeon @ 2.1 GHz, 1 core (VM), g++ 13 `-O2`, BS=64, best of 3 (1 run at N=2048). Speedup = naive / kernel.

| N | Naive (s) | Blocked (s) | Blocked speedup | Naive GFLOP/s | Blocked GFLOP/s |
|---|---|---|---|---|---|
| 256 | 0.020 | 0.012 | 1.7× | 1.7 | 2.8 |
| 512 | 0.314 | 0.096 | 3.3× | 0.9 | 2.8 |
| 1024 | 4.606 | 0.793 | **5.8×** | 0.5 | 2.7 |
| 2048 | 40.48 | 7.14 | **5.7×** | 0.4 | 2.4 |

Speedup grows with N because the naive kernel's working set falls out of cache while blocked stays cache-resident.
All kernels match naive exactly (max abs diff 0).

## Profiling
```bash
perf stat -e cache-references,cache-misses,L1-dcache-load-misses ./build/matmul 1024
perf record -g ./build/matmul 1024 && perf report
perf c2c record ./build/matmul 1024 && perf c2c report     # false-sharing hot lines
cmake -B build-pg -DENABLE_GPROF=ON && cmake --build build-pg
./build-pg/matmul 512 1 && gprof ./build-pg/matmul gmon.out
```
`gprof` only samples the main thread, so use `perf` for the threaded kernels.

## False Sharing
Two threads writing different variables on the same 64B line force it to bounce between cores' caches. `mul_threaded_false_sharing` shows this in the matmul itself; `mul_threaded` fixes it by giving each thread whole row bands. `false_sharing.cpp` isolates the effect with packed vs `alignas(64)` counters.

## Possible Improvements
AVX2/FMA vectorisation, register tiling, B-transpose packing, tuning BS per cache size.
