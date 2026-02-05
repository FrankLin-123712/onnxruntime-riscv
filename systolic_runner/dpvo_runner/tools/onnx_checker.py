import argparse
import subprocess
import sys

import onnxruntime as ort, numpy as np, onnx

""" Description
1. _stage_onnx_check

- Validates the model against the ONNX spec.
- onnx.checker.check_model(path) ensures the file is a valid ONNX model.
- onnx.checker.check_model(loaded) re-validates the in-memory graph (same checks, but after parsing).

2. _stage_shape_inference

- Runs ONNX shape inference (onnx.shape_inference.infer_shapes).
- Verifies that all input shapes are fully inferable (no None dims).
- If any input dimension is unknown, it fails.

3. _stage_input_validation

- Validates all model inputs are tensor types with supported dtypes.
- Prints input names, dtypes, and shapes.

4. _stage_ort_run

- Loads the model with ONNX Runtime CPU and does a real inference using generated inputs.
- Catches runtime issues like unsupported ops/kernels, mismatched shapes, missing initializers, etc.

5. _stage_mobile_usability

- Runs ORT Mobile’s usability checker via the CLI.
- Reports which parts can be partitioned to mobile EPs (CoreML/NNAPI), unsupported ops, caveats, and performance guidance.
"""


def _tensor_shape_from_value_info(value_info):
    if not value_info.type.HasField("tensor_type"):
        return None
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims = []
    for d in tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append(None)
    return dims


def _numpy_dtype_from_onnx(elem_type):
    if elem_type == onnx.TensorProto.FLOAT:
        return np.float32
    if elem_type == onnx.TensorProto.FLOAT16:
        return np.float16
    if elem_type == onnx.TensorProto.DOUBLE:
        return np.float64
    if elem_type == onnx.TensorProto.INT8:
        return np.int8
    if elem_type == onnx.TensorProto.INT16:
        return np.int16
    if elem_type == onnx.TensorProto.INT32:
        return np.int32
    if elem_type == onnx.TensorProto.INT64:
        return np.int64
    if elem_type == onnx.TensorProto.UINT8:
        return np.uint8
    if elem_type == onnx.TensorProto.UINT16:
        return np.uint16
    if elem_type == onnx.TensorProto.UINT32:
        return np.uint32
    if elem_type == onnx.TensorProto.UINT64:
        return np.uint64
    if elem_type == onnx.TensorProto.BOOL:
        return np.bool_
    if elem_type == onnx.TensorProto.STRING:
        return np.object_
    if elem_type == onnx.TensorProto.COMPLEX64:
        return np.complex64
    if elem_type == onnx.TensorProto.COMPLEX128:
        return np.complex128
    if elem_type == onnx.TensorProto.BFLOAT16:
        try:
            return np.dtype("bfloat16")
        except TypeError as exc:
            raise ValueError("BFLOAT16 not supported by this NumPy build") from exc
    raise ValueError(f"Unsupported ONNX elem_type: {elem_type}")


def _stage_onnx_check(model_path):
    onnx.checker.check_model(model_path)
    print("PASS: onnx.checker.check_model(path)")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("PASS: onnx.checker.check_model(loaded)")
    return model


def _stage_shape_inference(model):
    inferred = onnx.shape_inference.infer_shapes(model)
    unknown_dims = []
    for vi in inferred.graph.input:
        shape = _tensor_shape_from_value_info(vi)
        if shape is None:
            unknown_dims.append((vi.name, "non-tensor-or-missing-shape"))
            continue
        missing = [i for i, d in enumerate(shape) if d is None]
        if missing:
            unknown_dims.append((vi.name, missing, shape))
    if unknown_dims:
        raise ValueError(f"Uninferable input dims: {unknown_dims}")
    print("PASS: onnx.shape_inference.infer_shapes (inputs inferable)")
    return inferred


def _input_specs_from_graph(inferred):
    specs = []
    for vi in inferred.graph.input:
        if not vi.type.HasField("tensor_type"):
            raise ValueError(f"Input {vi.name} is not a tensor type")
        tensor_type = vi.type.tensor_type
        np_dtype = _numpy_dtype_from_onnx(tensor_type.elem_type)
        shape = _tensor_shape_from_value_info(vi)
        if shape is None:
            raise ValueError(f"Input {vi.name} has no shape")
        specs.append({"name": vi.name, "shape": shape, "dtype": np_dtype})
    return specs


def _stage_input_validation(inferred):
    specs = _input_specs_from_graph(inferred)
    if not specs:
        raise ValueError("Model has no inputs")
    print(f"PASS: input validation ({len(specs)} inputs)")
    for spec in specs:
        print(f"  input: {spec['name']} dtype={spec['dtype']} shape={spec['shape']}")
    return specs


def _resolve_shape(shape, dim_values, unknown_dim_value):
    resolved = []
    for d in shape:
        if isinstance(d, int):
            resolved.append(d if d > 0 else 1)
        elif isinstance(d, str):
            resolved.append(dim_values.get(d, unknown_dim_value))
        else:
            resolved.append(unknown_dim_value)
    return resolved


def _make_random_tensor(shape, dtype):
    shape = tuple(shape)
    if dtype in (np.float16, np.float32, np.float64):
        return np.random.randn(*shape).astype(dtype)
    if dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        return np.random.randint(0, 2, size=shape, dtype=dtype)
    if dtype == np.bool_:
        return (np.random.rand(*shape) > 0.5).astype(np.bool_)
    if dtype == np.object_:
        return np.full(shape, "", dtype=np.object_)
    if dtype in (np.complex64, np.complex128):
        real = np.random.randn(*shape)
        imag = np.random.randn(*shape)
        return (real + 1j * imag).astype(dtype)
    raise ValueError(f"Unsupported numpy dtype for input generation: {dtype}")


def _stage_ort_run(model_path, input_specs, dim_values, unknown_dim_value, custom_op_libs):
    sess_options = ort.SessionOptions()
    for lib in custom_op_libs:
        sess_options.register_custom_ops_library(lib)
    sess = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
    feeds = {}
    for spec in input_specs:
        resolved_shape = _resolve_shape(spec["shape"], dim_values, unknown_dim_value)
        feeds[spec["name"]] = _make_random_tensor(resolved_shape, spec["dtype"])
    sess.run(None, feeds)
    print("PASS: onnxruntime InferenceSession + run")


def _stage_mobile_usability(model_path):
    subprocess.run(
        [sys.executable, "-m", "onnxruntime.tools.check_onnx_model_mobile_usability", model_path, "--log_level", "info"],
        check=True,
    )
    print("PASS: ORT mobile usability checker")


def _parse_dim_values(pairs):
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --dim-value '{pair}'. Use name=value.")
        name, value = pair.split("=", 1)
        out[name] = int(value)
    return out


def main():
    parser = argparse.ArgumentParser(description="ONNX model checks")
    parser.add_argument("model", help="Path to ONNX model")
    parser.add_argument(
        "--custom-op-lib",
        action="append",
        default=[],
        help="Path to custom ops shared library (.so). Can be provided multiple times.",
    )
    parser.add_argument(
        "--dim-value",
        action="append",
        default=[],
        help="Override symbolic dim values, e.g. --dim-value batch=1 --dim-value edges=16",
    )
    parser.add_argument(
        "--unknown-dim-value",
        type=int,
        default=1,
        help="Value to use for unknown (None) dims when generating inputs",
    )
    parser.add_argument(
        "--skip-ort-run",
        action="store_true",
        help="Skip ONNX Runtime session load/run (useful if custom ops aren't available).",
    )
    args = parser.parse_args()

    dim_values = _parse_dim_values(args.dim_value)

    model = _stage_onnx_check(args.model)
    inferred = _stage_shape_inference(model)
    input_specs = _stage_input_validation(inferred)
    if args.skip_ort_run:
        print("SKIP: onnxruntime InferenceSession + run")
    else:
        _stage_ort_run(args.model, input_specs, dim_values, args.unknown_dim_value, args.custom_op_lib)
    _stage_mobile_usability(args.model)


if __name__ == "__main__":
    main()
