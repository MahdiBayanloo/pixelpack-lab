#include "pixelpack.h"
#include <zlib.h>
#include <limits>
#include <initializer_list>

namespace {
constexpr size_t max_bytes = 256u * 1024u * 1024u;
bool valid(size_t w, size_t h, size_t c, size_t b) {
    if (!w || !h || !c || c > 4 || (b != 1 && b != 2 && b != 4)) return false;
    size_t n = 1;
    for (size_t x : {w, h, c, b}) { if (x > max_bytes / n) return false; n *= x; }
    return n <= max_bytes;
}
uint32_t load(const uint8_t* p, size_t b) {
    uint32_t v = 0;
    for (size_t j = 0; j < b; ++j) v |= uint32_t(p[j]) << (8*j);
    return v;
}
void store(uint8_t* p, uint32_t v, size_t b) {
    for (size_t j = 0; j < b; ++j) p[j] = uint8_t(v >> (8*j));
}
}
extern "C" int pp_transform(const uint8_t* src, uint8_t* dst, size_t w, size_t h, size_t c, size_t b, int inverse) {
    try {
        if (!src || !dst || src == dst || !valid(w,h,c,b) || (inverse != 0 && inverse != 1)) return -1;
        const size_t stride = w*c, n = stride*h;
        if (!inverse) {
            // Each prediction reads original input, so iterations are independent.
            #pragma omp parallel for schedule(static) if(n >= 65536)
            for (size_t i = 0; i < n; ++i) {
                uint32_t x = load(src+i*b,b), p = i >= stride ? load(src+(i-stride)*b,b) : 0;
                uint32_t r = b == 4 ? (x ^ p) : (x - p);
                for (size_t j = 0; j < b; ++j) dst[j*n+i] = uint8_t(r >> (8*j));
            }
        } else {
            // Rows depend on previous reconstructed row; columns/channels do not.
            #pragma omp parallel for schedule(static) if(n >= 65536)
            for (size_t col = 0; col < stride; ++col) {
                uint32_t p = 0;
                for (size_t row = 0; row < h; ++row) {
                    const size_t i = row*stride+col;
                    uint32_t r = 0;
                    for (size_t j = 0; j < b; ++j) r |= uint32_t(src[j*n+i]) << (8*j);
                    uint32_t x = b == 4 ? (r ^ p) : (r + p);
                    store(dst+i*b,x,b);
                    p = x; // modulo truncation in store is equivalent for 1/2-byte samples.
                }
            }
        }
        return 0;
    } catch (...) { return -3; }
}
extern "C" int pp_compress(const uint8_t* src, size_t n, uint8_t* dst, size_t* cap, int level) {
    if (!src || !dst || !cap || !n || n > max_bytes || level < 0 || level > 9 || *cap > std::numeric_limits<uLongf>::max()) return -1;
    uLongf out = *cap;
    int rc = compress2(dst,&out,src,n,level);
    *cap = out;
    return rc == Z_OK ? 0 : -2;
}
extern "C" int pp_decompress(const uint8_t* src, size_t n, uint8_t* dst, size_t expected) {
    if (!src || !dst || !n || !expected || expected > max_bytes || n > max_bytes+1048576) return -1;
    uLongf out = expected; uLong consumed = n;
    int rc = uncompress2(dst,&out,src,&consumed);
    return rc == Z_OK && out == expected && consumed == n ? 0 : -2;
}
extern "C" const char* pp_zlib_version() { return zlibVersion(); }
