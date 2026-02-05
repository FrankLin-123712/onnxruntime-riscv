import onnxruntime as ort, numpy as np, onnx
m = "onnx_models/feature_extractor.onnx"
onnx.checker.check_model(onnx.load(m))
sess = ort.InferenceSession(m, providers=["CPUExecutionProvider"])
x = np.random.randn(1,1,3,480,640).astype(np.float32)
sess.run(None, {sess.get_inputs()[0].name: x})