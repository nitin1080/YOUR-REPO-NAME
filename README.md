# High-Performance Matrix Multiplication (C++17)

Naive, cache-blocked and multithreaded N×N `double` matrix multiplication, written from scratch with no external libraries. Built to study cache locality, thread scaling and false sharing.

## Features
- **Naive** ijk kernel as the baseline
- **Cache-blocked** kernel: tiling plus ikj loop order, so all inner-loop accesses are unit-stride
- **Multithreaded** kernel: `std::thread`, each thread owns a disjoint band of C rows
- Correctness check of every kernel against the naive result
- False-sharing micro-benchmark and `perf` / `gprof` profiling support

## Structure
```
matmul.hpp          kernels: mul_naive, mul_blocked, mul_threaded
main.cpp            benchmark + correctness check
false_sharing.cpp   packed vs 64B-padded counters demo
CMakeLists.txt      Release build, -g for perf, optional gprof (-pg)
```

## Build & Run
```bash
cmake -B build && cmake --build build
./build/matmul [N=1024] [threads=hw] [block=64]
./build/false_sharing
```
No CMake: `g++ -std=c++17 -O2 -g -fno-omit-frame-pointer main.cpp -o matmul -pthread`

## How It Works
- **Naive:** `B[k][j]` is read with stride N, so nearly every access is a cache miss.
- **Blocked:** the matrices are split into BS×BS tiles (default 64). A tile of B stays in L1/L2 while it is reused across a whole block of rows. The ikj order makes `C[i][j]` and `B[k][j]` sequential in memory.
- **Threaded:** C is split into row bands whose boundaries are multiples of BS rows. Threads never write the same cache line, so there is no false sharing and no locks.

## Results
N = 1024, BS = 64, `-O2`, single-core VM:

| Kernel | Time (s) | GFLOP/s | Speedup |
|---|---|---|---|
| Naive | 4.52 | 0.48 | 1.00× |
| Blocked | 0.73 | 2.93 | **6.17×** |
| Threaded (`<T>` threads) | `<fill>` | `<fill>` | `<fill>` |

Blocking alone gives the 6× gain. Thread scaling needs a multi-core machine.

## Profiling
```bash
perf stat -e cache-references,cache-misses,L1-dcache-load-misses ./build/matmul 1024
perf record -g ./build/matmul 1024 && perf report
cmake -B build-pg -DENABLE_GPROF=ON && cmake --build build-pg
./build-pg/matmul 1024 && gprof ./build-pg/matmul gmon.out
```

## False Sharing Demo
`false_sharing` increments per-thread counters packed into one cache line vs `alignas(64)` padded. Packed counters bounce the line between cores and run noticeably slower on multi-core CPUs.

## Possible Improvements
AVX2/FMA vectorisation, register tiling, B-transpose packing, tuning BS per cache size.
