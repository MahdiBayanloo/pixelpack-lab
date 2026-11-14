# Design and engineering decisions

## Data contract and API

`pp_transform(src, dst, width, height, channels, itemsize, inverse)` consumes packed, contiguous HWC samples. Callers own non-overlapping, sufficiently sized buffers; equal input/output pointers are rejected. C++ does not own application memory and exceptions do not cross the C ABI. `pp_compress`/`pp_decompress` delegate to zlib and check result codes. Python ctypes provides the typed array/container layer. No CPython extension build system is needed.

The low-level C API is not safe for arbitrary caller pointers: as with many C interfaces, buffer lengths and valid ownership are caller obligations. The Python boundary validates shape, type, size and payload metadata. This is an educational prototype, not a hardened C ABI.

## Reversible prediction

For image sample `x[r,c,k]`, take `p=x[r-1,c,k]`, or zero on row 0. For uint8/uint16, store `(x-p) mod 2^bits`; inverse reconstructs `(residual+p) mod 2^bits`. C++ unsigned arithmetic and byte truncation implement this without signed overflow.

For float32 samples, interpret each sample as four bytes, assemble the bits in a uint32 and store `bits(x) XOR bits(p)`. The inverse uses the same XOR relation. There is no float arithmetic, so there is no new rounding error; every original bit is reconstructed. Tests use positive/negative zero, infinities, subnormals, ordinary values and two different NaN payloads.

For N samples of B bytes, byte shuffle maps sample byte `j` at sample `i` to `out[j*N+i]`. It is a permutation, not a lossy quantization. Little-endian serialization is explicit, avoiding unaligned native uint16/uint32 dereferences and host-endian assumptions.

## Parallel dependency analysis

- Forward: reads the original prior row, never a newly written residual. Every output sample is independent. CPU OpenMP splits samples; CUDA assigns one thread per sample with a tail guard.
- Inverse: each column/channel has a vertical reconstruction dependency. CPU and CUDA split column/channel streams; each stream walks rows in order. Launching one independent inverse thread per sample would be incorrect without a prefix-scan strategy.
- CPU OpenMP activates only at 65,536 samples to avoid thread overhead on tiny arrays. It is an engineering heuristic, not a benchmark-proven optimal threshold.
- Shuffling creates strided byte-plane writes; optional further profiling should inspect memory throughput and access coalescing. No performance claims about the CUDA kernel are made before execution.

## Compression and selection

Three modes use the same zlib level 6 and container header:

| Mode | Input to zlib | Cost |
|---|---|---|
| raw | Original canonical bytes | One compression pass |
| predict | Predicted and shuffled bytes | Transform plus one pass |
| adaptive | Both, keep smaller | Transform plus two passes |

The adaptive selector chooses raw on ties. It cannot exceed the raw candidate's container size, but has extra time and peak-memory costs. The 256 MiB decoded-size limit is **not** a total memory limit: two candidates, transformed bytes and copies coexist. This is a size-oriented prototype, not a memory-optimal streaming codec.

## Container layout

Little-endian Python struct `<4sBBBBIIQI`, 28 bytes:

| Field | Bytes | Meaning |
|---|---:|---|
| magic | 4 | PPL1 |
| version | 1 | 1 |
| dtype | 1 | 1=uint8, 2=uint16, 3=float32 |
| channels | 1 | 1..4 |
| predictor | 1 | 0=raw, 1=predict+shuffle |
| width | 4 | Positive |
| height | 4 | Positive |
| payload length | 8 | Must match remaining bytes exactly |
| CRC32 | 4 | Original canonical pixel bytes |

Decoder rejects unknown versions/types, invalid channels/dimensions, oversized decoded arrays, truncated/trailing payloads, zlib failures and reconstructed CRC mismatch. CRC32 is accidental-corruption detection, not authentication. It covers raw pixels, not the header: malicious same-size shape changes are not cryptographically detected. Do not use this format as a security boundary.

## Measurement method

Seed 2026569. Ten synthetic cases and one UCI optical-digits mosaic. Canonical input hashes are recorded. Every mode is warmed once; seven encode/decode repetitions use `perf_counter_ns`, with full-array byte comparisons outside the timed decode region. Encode timing includes normalization, allocation, C ABI, transform/compression and container creation. Decode timing includes header/size checks, allocation, inflate, inverse, CRC and array reconstruction.

The baseline is raw bytes plus the same container header and zlib level, not a straw-man uncompressed file and not an optimized image codec. The report includes every dataset. Aggregate bytes are weighted by actual output size, not an unweighted average of percentages. Shared-container timing is noisy; min/median/max encode timings are recorded. Repeated deterministic measurements do not establish statistical significance or performance across machines.

## Correctness evidence

The NumPy oracle computes the transform independently and compares all output bytes. This catches an important blind spot: forward/inverse code can share the same bug and still round-trip. Additional tests cover endian normalization, non-contiguous views, modular wraparound, special float bits, all header truncation positions, invalid fields, CRC mismatch, corruption, trailing bytes, cross-decoding with Python zlib, and PNG/TIFF pixel import/export through Pillow. Native ASan/UBSan exercises 81 dtype/shape/channel combinations. It is not a full fuzz campaign.

## What would count as a real improvement next?

1. Reproduce the CPU report locally; record OS/compiler/thread settings.
2. Profile before changing code. Establish serial/OpenMP scaling with fixed image sizes and controlled threads.
3. Compile and run CUDA correctness and Compute Sanitizer; save raw outputs.
4. Measure H2D transfer, forward kernel, host DEFLATE and D2H separately and together.
5. Evaluate chunking or a sample-based selector against the current two-pass method on a held-out dataset. Measure misses as well as savings.
6. Report actual baseline versus changed results with source revisions. Until then, these are proposed experiments, not delivered optimizations.
