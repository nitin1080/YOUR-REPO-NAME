// Micro-benchmark: per-thread counters packed together vs padded to 64B.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

struct Packed { volatile long v; };                 // 8B -> 8 counters per line
struct alignas(64) Padded { volatile long v; };     // 1 counter per line

template <class T>
double run(unsigned nt, long iters) {
    std::vector<T> c(nt);
    for (auto& x : c) x.v = 0;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < nt; ++t)
        ts.emplace_back([&, t] { for (long i = 0; i < iters; ++i) c[t].v++; });
    for (auto& th : ts) th.join();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    unsigned nt = std::max(2u, std::thread::hardware_concurrency());
    const long iters = 100'000'000;
    double a = run<Packed>(nt, iters), b = run<Padded>(nt, iters);
    std::printf("threads=%u  packed(false sharing)=%.3fs  padded(64B)=%.3fs  slowdown=%.2fx\n",
                nt, a, b, a / b);
}
