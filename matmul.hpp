#pragma once
#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

// Row-major N x N matrix of doubles, 64-byte aligned rows not required
// (we align the whole buffer and partition threads on row blocks).
struct Matrix {
    size_t n;
    std::vector<double> d;
    explicit Matrix(size_t n_) : n(n_), d(n_ * n_, 0.0) {}
    double* operator[](size_t i) { return &d[i * n]; }
    const double* operator[](size_t i) const { return &d[i * n]; }
    void zero() { std::fill(d.begin(), d.end(), 0.0); }
};

// 1) Naive ijk: B is walked column-wise -> stride-N access, cache-hostile.
inline void mul_naive(const Matrix& A, const Matrix& B, Matrix& C) {
    const size_t n = A.n;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0;
            for (size_t k = 0; k < n; ++k) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
}

// 2) Cache-blocked (tiled), ikj inner order: all three inner accesses are
//    unit-stride and a BSxBS tile of B stays resident in L1/L2.
inline void mul_blocked_rows(const Matrix& A, const Matrix& B, Matrix& C,
                             size_t i_begin, size_t i_end, size_t BS = 64) {
    const size_t n = A.n;
    for (size_t ii = i_begin; ii < i_end; ii += BS)
        for (size_t kk = 0; kk < n; kk += BS)
            for (size_t jj = 0; jj < n; jj += BS) {
                const size_t imax = std::min(ii + BS, i_end);
                const size_t kmax = std::min(kk + BS, n);
                const size_t jmax = std::min(jj + BS, n);
                for (size_t i = ii; i < imax; ++i)
                    for (size_t k = kk; k < kmax; ++k) {
                        const double a = A[i][k];
                        for (size_t j = jj; j < jmax; ++j)
                            C[i][j] += a * B[k][j];
                    }
            }
}

inline void mul_blocked(const Matrix& A, const Matrix& B, Matrix& C, size_t BS = 64) {
    C.zero();
    mul_blocked_rows(A, B, C, 0, A.n, BS);
}

// 3) Multithreaded blocked: each thread owns a disjoint band of C rows.
//    Band boundaries are rounded to BS rows (>= 1 cache line of doubles = 8),
//    so no two threads ever write the same cache line -> no false sharing.
inline void mul_threaded(const Matrix& A, const Matrix& B, Matrix& C,
                         unsigned nthreads, size_t BS = 64) {
    const size_t n = A.n;
    C.zero();
    if (nthreads == 0) nthreads = 1;
    const size_t blocks = (n + BS - 1) / BS;
    std::vector<std::thread> ts;
    size_t b0 = 0;
    for (unsigned t = 0; t < nthreads; ++t) {
        size_t cnt = blocks / nthreads + (t < blocks % nthreads ? 1 : 0);
        size_t i0 = b0 * BS, i1 = std::min((b0 + cnt) * BS, n);
        b0 += cnt;
        if (i0 >= i1) continue;
        ts.emplace_back([&, i0, i1] { mul_blocked_rows(A, B, C, i0, i1, BS); });
    }
    for (auto& th : ts) th.join();
}

// 4) Deliberately BAD threaded kernel (for false-sharing analysis only):
//    columns are interleaved across threads (thread t computes j = t, t+T, ...),
//    so neighbouring threads write adjacent C[i][j] inside the same 64B line.
inline void mul_threaded_false_sharing(const Matrix& A, const Matrix& B, Matrix& C,
                                       unsigned nthreads) {
    const size_t n = A.n;
    C.zero();
    if (nthreads == 0) nthreads = 1;
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < nthreads; ++t)
        ts.emplace_back([&, t] {
            for (size_t i = 0; i < n; ++i)
                for (size_t j = t; j < n; j += nthreads) {
                    double* c = &C[i][j];              // written directly in the loop
                    for (size_t k = 0; k < n; ++k) *c += A[i][k] * B[k][j];
                }
        });
    for (auto& th : ts) th.join();
}
