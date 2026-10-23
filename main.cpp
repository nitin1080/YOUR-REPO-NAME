#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include "matmul.hpp"

using Clock = std::chrono::steady_clock;
template <class F> double time_s(F&& f) {
    auto t0 = Clock::now(); f();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double max_diff(const Matrix& a, const Matrix& b) {
    double m = 0;
    for (size_t i = 0; i < a.d.size(); ++i) m = std::max(m, std::fabs(a.d[i] - b.d[i]));
    return m;
}

int main(int argc, char** argv) {
    size_t n = argc > 1 ? std::stoul(argv[1]) : 1024;
    unsigned nt = argc > 2 ? std::stoul(argv[2]) : std::max(1u, std::thread::hardware_concurrency());
    size_t bs = argc > 3 ? std::stoul(argv[3]) : 64;

    Matrix A(n), B(n), Cn(n), Cb(n), Ct(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> U(-1, 1);
    for (auto& x : A.d) x = U(rng);
    for (auto& x : B.d) x = U(rng);

    double tn = time_s([&] { mul_naive(A, B, Cn); });
    double tb = time_s([&] { mul_blocked(A, B, Cb, bs); });
    double tt = time_s([&] { mul_threaded(A, B, Ct, nt, bs); });

    std::printf("N=%zu  block=%zu  threads=%u\n", n, bs, nt);
    std::printf("%-10s %9s %9s %10s\n", "kernel", "time(s)", "GFLOP/s", "speedup");
    auto row = [&](const char* name, double t) {
        std::printf("%-10s %9.3f %9.2f %9.2fx\n", name, t, 2.0 * n * n * n / t / 1e9, tn / t);
    };
    row("naive", tn); row("blocked", tb); row("threaded", tt);
    std::printf("max|naive-blocked|=%.2e  max|naive-threaded|=%.2e\n",
                max_diff(Cn, Cb), max_diff(Cn, Ct));
    return 0;
}
