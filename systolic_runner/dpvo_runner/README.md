# DPVO Runner (systolic)

Lightweight C++ runner that mirrors `DPVO/demo.py`, split into:
- Feature extractor (ONNX session)
- DPVO state manager (C++ glue)
- Correlation (C++ CPU implementation placeholder)
- Heads block: 1D_Conv / Soft-agg / Transition / Factor_head (ONNX session)
- Bundle adjustment (C++ CPU implementation placeholder)

Status: scaffold only. Math kernels for correlation and bundle adjustment are TODO; the runner currently calls stubs so it can build and be wired to real implementations later.

Usage (example):
```
dpvo_runner \
  --feature_model models/feature_extractor.onnx \
  --head_model models/heads.onnx \
  --sequence_dir /path/to/frames \
  --calib K.txt \
  --stride 1 --skip 0 \
  -x 0 -O 1
```

Build:
```
./build.sh <path-to-ort-build> <root-path>
```

Notes:
- Providers are ordered [Systolic, CPU]; unsupported ops fall back to CPU automatically.
- Replace the stub correlation and BA functions in `src/dpvo_runner.cpp` with working implementations.
