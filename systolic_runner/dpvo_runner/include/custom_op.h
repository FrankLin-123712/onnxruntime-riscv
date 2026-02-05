#ifndef DPVO_CUSTOM_OP_H_
#define DPVO_CUSTOM_OP_H_

#include "onnxruntime_cxx_api.h"

namespace dpvo {

struct ScatterMaxKernel {
  ScatterMaxKernel(const OrtApi& api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* ctx);

 private:
  const OrtApi& api_;
  Ort::CustomOpApi ort_;
};

struct ScatterMaxOp : Ort::CustomOpBase<ScatterMaxOp, ScatterMaxKernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  const char* GetExecutionProviderType() const;
};

}  // namespace dpvo

#endif  // DPVO_CUSTOM_OP_H_
