// Usage: matmul [N=1024] [threads=hw] [block=64] [--csv] [--skip-naive-above=N]
#include <chrono>
#include <cmath>
#include <cstdio>
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
// best of `reps` runs, to reduce noise
template <class F> double best(int reps, F&& f) {
    double b = 1e30;
    for (int r = 0; r < reps; ++r) b = std::min(b, time_s(f));
    return b;
}

int main(int argc, char** argv) {
    size_t n = 1024; unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    size_t bs = 64; bool csv = false; int reps = 3;
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv") csv = true;
        else if (a.rfind("--reps=", 0) == 0) reps = std::stoi(a.substr(7));
        else if (pos == 0) { n = std::stoul(a); ++pos; }
        else if (pos == 1) { nt = std::stoul(a); ++pos; }
        else if (pos == 2) { bs = std::stoul(a); ++pos; }
    }

    Matrix A(n), B(n), Cn(n), Cb(n), Ct(n), Cf(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> U(-1, 1);
    for (auto& x : A.d) x = U(rng);
    for (auto& x : B.d) x = U(rng);

    double tn = best(reps, [&] { mul_naive(A, B, Cn); });
    double tb = best(reps, [&] { mul_blocked(A, B, Cb, bs); });
    double tt = best(reps, [&] { mul_threaded(A, B, Ct, nt, bs); });
    double tf = best(reps, [&] { mul_threaded_false_sharing(A, B, Cf, nt); });
    double err = std::max({max_diff(Cn, Cb), max_diff(Cn, Ct), max_diff(Cn, Cf)});

    const double fl = 2.0 * n * n * n;
    if (csv) {
        std::printf("%zu,%u,%zu,%.4f,%.4f,%.4f,%.4f,%.2e\n", n, nt, bs, tn, tb, tt, tf, err);
        return 0;
    }
    std::printf("N=%zu  block=%zu  threads=%u  (best of %d)\n", n, bs, nt, reps);
    std::printf("%-22s %9s %9s %10s\n", "kernel", "time(s)", "GFLOP/s", "vs naive");
    auto row = [&](const char* nm, double t) {
        std::printf("%-22s %9.3f %9.2f %9.2fx\n", nm, t, fl / t / 1e9, tn / t);
    };
    row("naive", tn); row("blocked", tb);
    row("threaded (row bands)", tt); row("threaded (false shar.)", tf);
    std::printf("max abs diff vs naive: %.2e\n", err);
}
