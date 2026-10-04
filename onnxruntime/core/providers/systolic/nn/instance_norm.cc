// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef SYSTOLIC_FP32
#include "core/common/replay_profile.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/systolic_mlas.h"
#include "core/providers/cpu/nn/instance_norm_nhwc.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/providers/systolic/systolic_fwd.h"

namespace onnxruntime {
namespace systolic {

class InstanceNormNhwc final : public OpKernel {
 public:
  explicit InstanceNormNhwc(const OpKernelInfo& info)
      : OpKernel(info),
        epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-5f)),
        relu_(info.GetAttrOrDefault<int64_t>("relu", 0) != 0),
        mode_(static_cast<const SystolicExecutionProvider*>(info.GetExecutionProvider())
                  ->GetAcceleratorMode()) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const auto* scale = context->Input<Tensor>(1);
    const auto* bias = context->Input<Tensor>(2);
    const auto& shape = input->Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() == 4,
                      "InstanceNormalization_nhwc requires rank-4 NHWC input");
    const int64_t channels = shape.GetDims()[3];
    ORT_RETURN_IF_NOT(scale->Shape().NumDimensions() == 1 && scale->Shape().Size() == channels,
                      "InstanceNormalization_nhwc scale must have shape [C]");
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1 && bias->Shape().Size() == channels,
                      "InstanceNormalization_nhwc bias must have shape [C]");
    auto* output = context->Output(0, shape);
    if (shape.Size() == 0) return Status::OK();
    const int64_t batches = shape.GetDims()[0];
    const int64_t spatial = shape.Slice(1, 3).Size();
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    Tensor workspace(DataTypeImpl::GetType<float>(), TensorShape({3, channels}), allocator);
    if (SystolicInstanceNormNhwc(mode_, input->Data<float>(), scale->Data<float>(),
                                 bias->Data<float>(), output->MutableData<float>(),
                                 batches, spatial, channels, epsilon_, relu_,
                                 workspace.MutableData<float>(), 3 * channels)) {
      return Status::OK();
    }
    // Mode 0/1, other channel counts, and unavailable hardware use exactly
    // the same reference as the CPU provider, with no RoCC instructions.
    ort_replay::Scope profile("kernel", "instance_norm.nhwc_cpu");
    InstanceNormNhwcCpu(input->Data<float>(), scale->Data<float>(), bias->Data<float>(),
                        output->MutableData<float>(), batches, spatial, channels,
                        epsilon_, relu_, workspace.MutableData<float>());
    return Status::OK();
  }

 private:
  float epsilon_;
  bool relu_;
  char mode_;
};

ONNX_OPERATOR_KERNEL_EX(
    InstanceNormalization_nhwc, kOnnxDomain, 1, kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), InstanceNormNhwc);

}  // namespace systolic
}  // namespace onnxruntime
#endif  // SYSTOLIC_FP32
