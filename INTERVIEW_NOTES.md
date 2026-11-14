# Review before putting this project in an application

The initial code and measurements were produced with an AI coding assistant. You should be able to explain the following from the actual source, not memorize a sales pitch.

1. **What did you personally do?** At delivery: chose the application goal and commissioned the prototype. Record your subsequent local review, reproduction, changes and GPU runs separately. Do not claim you wrote every initial line.
2. **What is the algorithm?** Vertical modular prediction (uint8/16) or XOR (float32 bits), byte-plane shuffling, then zlib DEFLATE. The novel-to-you work is the integration and experiment, not inventing DEFLATE.
3. **Why is forward parallel but inverse different?** Forward reads original prior-row samples; inverse needs a reconstructed predecessor. Independent column/channel streams avoid data races.
4. **Why not call it JPEG/TIFF implementation?** Pillow handles those file formats. Our format stores canonical pixels; it does not preserve file metadata or implement those codecs.
5. **What improved?** Compressed size on certain structured arrays and the fixed mixed suite. Not overall speed: adaptive mode usually encodes slower. Digits/random controls did not improve in size.
6. **What does 33.33% mean?** `(4,772,254 - 3,181,673) / 4,772,254` across these 11 particular containers, using the same zlib level and header. It is not a general compression percentage or GPU speedup.
7. **How did you verify generated code?** The delivered verification uses an independent NumPy oracle, exact byte reconstruction, malformed-input tests, zlib cross-decoding and native sanitizers. After you reproduce them, explain what you inspected and changed yourself.
8. **What is unverified?** CUDA compile/runtime and performance; macOS CI until publication; production safety; broad performance on real image corpora.
9. **Why float32 XOR?** It preserves bit patterns, including -0 and NaN payloads. That is not a numerical solver or an error-analysis result.
10. **What would you change first?** Profile the current implementation; measure a chunked/sample-based selector to reduce adaptive dual-pass time. Treat that as future work until implemented and measured.

## Your review log

Fill this from real actions:

- Local date, machine and compiler:
- Tests reproduced (paste summary):
- Files/functions you reviewed:
- Defect or limitation you identified:
- Change you implemented and why:
- Before/after measurement with source revisions:
- Actual GPU run, if any:

Keep this log honest; an interviewer may prefer a thoughtful limitation and fix to a large unexplained metric.
