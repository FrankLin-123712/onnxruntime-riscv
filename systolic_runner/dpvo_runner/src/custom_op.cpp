#include "custom_op.h"

#include <unordered_map>
#include <vector>

namespace dpvo {

ScatterMaxKernel::ScatterMaxKernel(const OrtApi& api, const OrtKernelInfo* /*info*/)
    : api_(api), ort_(api_) {}

void ScatterMaxKernel::Compute(OrtKernelContext* ctx) {
  const OrtValue* src = ort_.KernelContext_GetInput(ctx, 0);
  const OrtValue* idx = ort_.KernelContext_GetInput(ctx, 1);

  OrtTensorTypeAndShapeInfo* src_info = ort_.GetTensorTypeAndShape(src);
  OrtTensorTypeAndShapeInfo* idx_info = ort_.GetTensorTypeAndShape(idx);
  size_t src_rank = ort_.GetDimensionsCount(src_info);
  size_t idx_rank = ort_.GetDimensionsCount(idx_info);
  if (src_rank != idx_rank) {
    ORT_CXX_API_THROW("scatter_max: src/index rank mismatch", ORT_INVALID_ARGUMENT);
  }

  std::vector<int64_t> src_shape(src_rank);
  std::vector<int64_t> idx_shape(idx_rank);
  ort_.GetDimensions(src_info, src_shape.data(), src_shape.size());
  ort_.GetDimensions(idx_info, idx_shape.data(), idx_shape.size());
  ort_.ReleaseTensorTypeAndShapeInfo(src_info);
  ort_.ReleaseTensorTypeAndShapeInfo(idx_info);

  size_t total = 1;
  for (auto d : src_shape) total *= static_cast<size_t>(d);
  const float* src_data = ort_.GetTensorData<float>(src);
  const int64_t* idx_data = ort_.GetTensorData<int64_t>(idx);

  // Flatten: compute max per index value, broadcast.
  std::unordered_map<int64_t, float> max_map;
  std::unordered_map<int64_t, int64_t> arg_map;
  for (size_t i = 0; i < total; ++i) {
    int64_t key = idx_data[i];
    auto it = max_map.find(key);
    if (it == max_map.end() || src_data[i] > it->second) {
      max_map[key] = src_data[i];
      arg_map[key] = static_cast<int64_t>(i);
    }
  }

  OrtValue* out_val = ort_.KernelContext_GetOutput(ctx, 0, src_shape.data(), src_shape.size());
  OrtValue* out_arg = ort_.KernelContext_GetOutput(ctx, 1, src_shape.data(), src_shape.size());
  float* out_data = ort_.GetTensorMutableData<float>(out_val);
  int64_t* out_arg_data = ort_.GetTensorMutableData<int64_t>(out_arg);

  for (size_t i = 0; i < total; ++i) {
    int64_t key = idx_data[i];
    out_data[i] = max_map[key];
    out_arg_data[i] = arg_map[key];
  }
}

void* ScatterMaxOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  return new ScatterMaxKernel(api, info);
}

const char* ScatterMaxOp::GetName() const {
  return "scatter_max";
}

size_t ScatterMaxOp::GetInputTypeCount() const {
  return 2;
}

ONNXTensorElementDataType ScatterMaxOp::GetInputType(size_t index) const {
  return index == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

size_t ScatterMaxOp::GetOutputTypeCount() const {
  return 2;
}

ONNXTensorElementDataType ScatterMaxOp::GetOutputType(size_t index) const {
  return index == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

const char* ScatterMaxOp::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
}

}  // namespace dpvo
