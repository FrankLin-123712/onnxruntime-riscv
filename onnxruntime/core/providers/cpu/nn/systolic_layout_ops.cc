// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CPU-only operators introduced by the Systolic graph transformer. These
// preserve NHWC buffers and combine adjacent elementwise work; they do not
// issue accelerator instructions.
#ifdef USE_SYSTOLIC
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/nn/instance_norm_nhwc.h"

#include <cmath>

namespace onnxruntime {

class AddRelu final : public OpKernel {
 public:
  explicit AddRelu(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    // Match Add's multidirectional broadcasting and Relu's Eigen maximum,
    // while writing each output element only once.
    ProcessBroadcastSpanFuncs funcs{
        [](BroadcastHelper& helper) {
          helper.OutputEigen<float>() =
              (helper.ScalarInput0<float>() + helper.EigenInput1<float>().array()).cwiseMax(0.0f);
        },
        [](BroadcastHelper& helper) {
          helper.OutputEigen<float>() =
              (helper.EigenInput0<float>().array() + helper.ScalarInput1<float>()).cwiseMax(0.0f);
        },
        [](BroadcastHelper& helper) {
          helper.OutputEigen<float>() =
              (helper.EigenInput0<float>() + helper.EigenInput1<float>()).cwiseMax(0.0f);
        }};
    UntypedBroadcastTwo(*context, funcs, 2.0);
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    AddRelu, 1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    AddRelu);

class InstanceNormNhwc final : public OpKernel {
 public:
  explicit InstanceNormNhwc(const OpKernelInfo& info)
      : OpKernel(info),
        epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-5f)),
        relu_(info.GetAttrOrDefault<int64_t>("relu", 0) != 0) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const auto* scale = context->Input<Tensor>(1);
    const auto* bias = context->Input<Tensor>(2);
    const auto& shape = input->Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() == 4,
                      "InstanceNormalization_nhwc requires rank-4 NHWC input");
    const int64_t channels = shape.GetDims()[3];
    ORT_RETURN_IF_NOT(scale->Shape().NumDimensions() == 1 &&
                          scale->Shape().Size() == channels,
                      "InstanceNormalization_nhwc scale must have shape [C]");
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1 &&
                          bias->Shape().Size() == channels,
                      "InstanceNormalization_nhwc bias must have shape [C]");
    auto* output = context->Output(0, shape);
    if (shape.Size() == 0) return Status::OK();

    const int64_t batches = shape.GetDims()[0];
    const int64_t spatial = shape.Slice(1, 3).Size();
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    Tensor workspace(DataTypeImpl::GetType<float>(), TensorShape({3, channels}), allocator);
    InstanceNormNhwcCpu(input->Data<float>(), scale->Data<float>(), bias->Data<float>(),
                        output->MutableData<float>(), batches, spatial, channels,
                        epsilon_, relu_, workspace.MutableData<float>());
    return Status::OK();
  }

 private:
  float epsilon_;
  bool relu_;
};

ONNX_CPU_OPERATOR_KERNEL(
    InstanceNormalization_nhwc, 1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    InstanceNormNhwc);

}  // namespace onnxruntime
#endif  // USE_SYSTOLIC
