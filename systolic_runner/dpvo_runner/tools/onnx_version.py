"""Small helper to inspect an ONNX file's metadata.

Usage:
  python verify_onnxmodel.py               # default exported_models/feature_extractor.onnx
  python verify_onnxmodel.py path/to/model.onnx
"""

import sys
import onnx

path = sys.argv[1] if len(sys.argv) > 1 else "exported_models/feature_extractor.onnx"
m = onnx.load(path)
print(f"Model: {path}")
print("IR version:", m.ir_version)
print("Producer:", m.producer_name, m.producer_version)
print("Opset imports:", {i.domain or 'ai.onnx': i.version for i in m.opset_import})

# Basic structural check so we fail fast if the file is corrupt.
onnx.checker.check_model(m)
print("onnx.checker: OK")