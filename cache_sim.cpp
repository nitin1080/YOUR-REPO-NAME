// Replays the exact load/store address stream of each loop nest through a
// set-associative LRU cache model and reports miss rates.
// Usage: cache_sim [N=256] [BS=32] [cacheKB=32] [ways=8] [line=64]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Cache {
    size_t sets, ways, line; uint64_t hits = 0, misses = 0;
    std::vector<uint64_t> tag; std::vector<uint64_t> lru; uint64_t clk = 0;
    Cache(size_t kb, size_t w, size_t l) : sets(kb * 1024 / (w * l)), ways(w), line(l),
        tag(sets * w, ~0ull), lru(sets * w, 0) {}
    void access(uint64_t addr) {
        uint64_t ln = addr / line; size_t s = ln % sets; uint64_t* t = &tag[s * ways];
        uint64_t* r = &lru[s * ways]; ++clk;
        size_t victim = 0;
        for (size_t w = 0; w < ways; ++w) {
            if (t[w] == ln) { r[w] = clk; ++hits; return; }
            if (r[w] < r[victim]) victim = w;
        }
        ++misses; t[victim] = ln; r[victim] = clk;
    }
    void reset() { hits = misses = clk = 0; std::fill(tag.begin(), tag.end(), ~0ull);
                   std::fill(lru.begin(), lru.end(), 0); }
};

int main(int argc, char** argv) {
    size_t n = argc > 1 ? atoi(argv[1]) : 256, bs = argc > 2 ? atoi(argv[2]) : 32;
    size_t kb = argc > 3 ? atoi(argv[3]) : 32, ways = argc > 4 ? atoi(argv[4]) : 8,
           line = argc > 5 ? atoi(argv[5]) : 64;
    const uint64_t Ab = 0, Bb = n * n * 8, Cb = 2 * n * n * 8;
    auto A = [&](size_t i, size_t k) { return Ab + (i * n + k) * 8; };
    auto B = [&](size_t k, size_t j) { return Bb + (k * n + j) * 8; };
    auto C = [&](size_t i, size_t j) { return Cb + (i * n + j) * 8; };
    Cache c(kb, ways, line);
    std::printf("N=%zu  BS=%zu  cache=%zuKB %zu-way %zuB lines\n", n, bs, kb, ways, line);
    auto report = [&](const char* nm) {
        double tot = c.hits + c.misses;
        std::printf("%-8s accesses=%.0f  misses=%llu  miss rate=%.2f%%\n", nm, tot,
                    (unsigned long long)c.misses, 100.0 * c.misses / tot);
    };
    // naive ijk
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
        for (size_t k = 0; k < n; ++k) { c.access(A(i, k)); c.access(B(k, j)); }
        c.access(C(i, j)); }
    report("naive"); c.reset();
    // blocked ikj
    for (size_t ii = 0; ii < n; ii += bs) for (size_t kk = 0; kk < n; kk += bs)
      for (size_t jj = 0; jj < n; jj += bs)
        for (size_t i = ii; i < ii + bs && i < n; ++i)
          for (size_t k = kk; k < kk + bs && k < n; ++k) {
            c.access(A(i, k));
            for (size_t j = jj; j < jj + bs && j < n; ++j) {
                c.access(B(k, j)); c.access(C(i, j)); c.access(C(i, j)); } }
    report("blocked");
}
