# DPVO Runner (systolic)

Lightweight C++ runner intended to mirror `DPVO/demo.py`.

Current pipeline:
- Feature extractor ONNX (`feature_extractor.onnx`)
- Patch/state management in C++
- Correlation: CPU approximation (needs altcorr equivalent)
- Update block ONNX (`update_block.onnx`) — uses custom op `dpvo::scatter_max`
- Bundle adjustment: CPU placeholder (needs fastba/Ceres)

Status: control flow mirrors DPVO; correlation and BA are still placeholders, so poses will not yet match Python.

## Build
```
./build.sh [--config=Release] [--use_hwacha] [--for_firesim] [--enable_training]
```
Produces `dpvo_runner` linked with ONNX Runtime and the custom `dpvo::scatter_max` CPU kernel.

## Run
```
dpvo_runner \
  --feature_model onnx_models/feature_extractor.onnx \
  --update_model  onnx_models/update_block.onnx \
  --sequence_dir /path/to/frames \
  --calib K.txt \
  --stride 1 --skip 0 \
  -x 0 -O 1 --timeit
```

## Exporting models
Use `tools/export_models.py` (or `export2onnx.sh` in the DPVO repo) with your `dpvo.pth`:
```
python tools/export_models.py --weights /root/projects/DPVO/dpvo.pth --out onnx_models
```

## TODO for parity
- Replace CPU corr with altcorr-equivalent kernel.
- Replace BA placeholder with fastba/Ceres and point-cloud updates.
- Patch selection, motion model, keyframing, loop-closure refinements.
