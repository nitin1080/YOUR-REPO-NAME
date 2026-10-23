# High-Performance Matrix Multiplication (C++17)

Naive, cache-blocked and multithreaded N×N double matmul, no external libraries.

## Kernels (`matmul.hpp`)
| kernel | idea |
|---|---|
| `mul_naive` | ijk loop; B accessed with stride N → cache misses |
| `mul_blocked` | BS×BS tiling + ikj order; all inner accesses unit-stride, tile stays in L1/L2 |
| `mul_threaded` | blocked kernel, each `std::thread` owns a disjoint band of C rows (multiple of BS rows) → no shared cache lines written |

## Build & run
    cmake -B build && cmake --build build
    ./build/matmul [N=1024] [threads=hw] [block=64]
    ./build/false_sharing
    # without cmake:
    g++ -std=c++17 -O2 -g -fno-omit-frame-pointer main.cpp -o matmul -pthread

## Profiling
    perf stat -e cache-references,cache-misses,L1-dcache-load-misses ./build/matmul 1024
    perf record -g ./build/matmul 1024 && perf report
    cmake -B build-pg -DENABLE_GPROF=ON && cmake --build build-pg
    ./build-pg/matmul 1024 && gprof ./build-pg/matmul gmon.out

## False sharing
`false_sharing.cpp`: per-thread counters packed in one cache line vs `alignas(64)` padded.
`mul_threaded` avoids it by partitioning C on row bands rounded to BS rows (≥ 8 doubles = 64B).

## Benchmarks (fill in from your machine)
| N | naive (s) | blocked (s) | threaded (s) | blocked speedup | threaded speedup |
|---|---|---|---|---|---|
| 512 | | | | | |
| 1024 | | | | | |
| 2048 | | | | | |

Reference run (1-core sandbox, N=1024, BS=64): naive 4.52s, blocked 0.73s → **6.17×**.
Thread scaling and false-sharing numbers need a multi-core machine.
