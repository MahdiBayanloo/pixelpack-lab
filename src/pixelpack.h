#pragma once
#include <cstddef>
#include <cstdint>

// Packed HWC, little-endian samples, itemsize in {1,2,4}. No overlap allowed.
// 1/2-byte data: vertical modular difference. 4-byte data: vertical bitwise XOR.
// All modes apply byte-plane shuffling after prediction. Inverse is bit-exact.
// Return 0 on success; -1 invalid argument, -2 zlib error, -3 exception.
extern "C" {
int pp_transform(const uint8_t*, uint8_t*, size_t w, size_t h, size_t c, size_t itemsize, int inverse);
int pp_compress(const uint8_t*, size_t n, uint8_t*, size_t* capacity, int level);
int pp_decompress(const uint8_t*, size_t n, uint8_t*, size_t expected);
const char* pp_zlib_version();
}
