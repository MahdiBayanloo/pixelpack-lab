# Measured CPU benchmark

Date: 14 November 2026. Re-run `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python tools/benchmark.py`.

All figures below were measured on this run. Full distributions (min/median/max encode times), dataset hashes and environment are in `cpu_results.json`. Each mode is warmed once and repeated seven times. All decodes are compared byte-for-byte.

Baseline = the same container and zlib level 6 without the predictor. Adaptive tries both representations and keeps the smaller; its encoding time includes BOTH compression passes. Savings include the identical container header. This is not a comparison against PNG, TIFF, nvCOMP or nvTIFF.

| Dataset | Input bytes | Baseline bytes | Adaptive bytes | Size reduction | Baseline encode ms | Adaptive encode ms | Adaptive decode ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| u8_constant | 262656 | 305 | 305 | 0.00% | 0.382 | 1.092 | 0.061 |
| u8_horizontal_ramp | 262656 | 2084 | 620 | 70.25% | 0.385 | 1.125 | 0.214 |
| u8_vertical_ramp | 262656 | 1347 | 307 | 77.21% | 0.406 | 1.161 | 0.216 |
| u8_rgb_gradient | 787968 | 376421 | 1544 | 99.59% | 4.445 | 6.447 | 0.615 |
| u8_random | 787968 | 788247 | 788247 | 0.00% | 8.711 | 18.321 | 0.091 |
| u16_smooth_field | 525312 | 521628 | 1268 | 99.76% | 6.044 | 7.139 | 0.281 |
| u16_noisy_field | 525312 | 509719 | 141727 | 72.20% | 8.631 | 15.893 | 0.561 |
| u16_random | 525312 | 525511 | 525511 | 0.00% | 5.802 | 11.811 | 0.053 |
| f32_smooth_field | 1050624 | 950857 | 626009 | 34.16% | 21.076 | 43.191 | 1.809 |
| f32_random_bits | 1050624 | 1050983 | 1050983 | 0.00% | 11.648 | 23.645 | 0.127 |
| uci_optical_digits_sheet | 115584 | 45152 | 45152 | 0.00% | 6.138 | 8.764 | 0.120 |

Across this fixed mixed suite: 4,772,254 baseline bytes versus 3,181,673 adaptive bytes (33.33% smaller). This aggregate is strongly dataset-dependent; it is not a production compression claim.

## Interpretation

Vertical prediction exploits similarity between adjacent rows. Shuffling groups residual byte positions before DEFLATE. Uniform random inputs act as negative controls: prediction has no dependable compression benefit. The adaptive selector prevents a larger payload than the raw candidate, but often makes encoding slower. The report retains these costs and all negative controls.

32-bit samples use bitwise XOR, not floating-point arithmetic. Consequently signed zero, infinity, subnormal values and NaN payload bits can round-trip exactly. This is bit-preservation evidence, not floating-point numerical-error analysis.

## Validation and limitations

15 Python unittest methods (including parameterized subcases); independent NumPy transform oracle; PNG/TIFF pixel import/export; malformed headers, truncation, trailing bytes and CRC mismatch. Separate native ASan/UBSan execution covers 81 shape/type combinations.

CUDA source provides forward and inverse transforms plus a CPU/GPU comparison harness. It has NOT been compiled or executed in this environment. No GPU performance, correctness or speedup result is claimed. DEFLATE remains CPU zlib; there is no GPU entropy coder. The decoder is bounded to 256 MiB; this is a research prototype, not an audited untrusted-input production library.

## Environment

```json
{
  "platform": "macOS-26.6.2-arm64-arm-64bit",
  "machine": "arm64",
  "python": "3.12.7",
  "numpy": "2.5.4",
  "zlib": "1.2.12",
  "compiler": "Apple clang version 21.0.0 (clang-2100.1.1.101)",
  "omp_num_threads": "unset",
  "build_command": "c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic  -fPIC -shared src/pixelpack.cpp -lz",
  "repeats": 7,
  "timer": "perf_counter_ns; medians, seven repetitions after one warm-up",
  "seed": 2026569,
  "cuda": "NOT compiled or run: no nvcc / NVIDIA GPU in authoring environment",
  "scope": "local execution environment; hardware was not reserved exclusively",
  "implementation_sha256": {
    "src/pixelpack.cpp": "67082b9c0e4503a8f44c8ee14b142f77a10dce144c5dcb25a6debdc930ef1d9a",
    "src/pixelpack.h": "7cebf1cc1eaee7bb59bcbb3927c5dc61bed0641dc45506f3032b90d5cff10b93",
    "pixelpack.py": "973a63c3a628482c24795e02dccc0d11b5df184225bd51c986eb16a847b93747"
  }
}
```
