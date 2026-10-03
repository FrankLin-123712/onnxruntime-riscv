// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef SYSTOLIC_FP32
#include "core/common/replay_profile.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/systolic_mlas.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/providers/systolic/systolic_fwd.h"

namespace onnxruntime {
namespace systolic {

template <bool Relu>
class FloatAdd final : public OpKernel {
 public:
  explicit FloatAdd(const OpKernelInfo& info)
      : OpKernel(info),
        mode_(static_cast<const SystolicExecutionProvider*>(info.GetExecutionProvider())
                  ->GetAcceleratorMode()) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* a = context->Input<Tensor>(0);
    const auto* b = context->Input<Tensor>(1);
    if (a->Shape() == b->Shape()) {
      auto* output = context->Output(0, a->Shape());
      const auto count = static_cast<size_t>(a->Shape().Size());
      if (count == 0) return Status::OK();
      // Flat contiguous residual tensors need no layout conversion or padding.
      // Only WS residual-add uses Gemmini; other modes retain a CPU reference.
      SystolicAdd(mode_, Relu, a->Data<float>(), 1.0f, b->Data<float>(), 1.0f,
                  output->MutableData<float>(), 1.0f, count);
      return Status::OK();
    }

    // Preserve ONNX multidirectional broadcasting without materializing full
    // expanded tensors or issuing tiny accelerator transfers for each span.
    ort_replay::Scope scope("kernel", "add.broadcast_cpu");
    if (scope.Active()) scope.Detail(Relu ? "fused_relu=1" : "fused_relu=0");
    ProcessBroadcastSpanFuncs funcs{
        [](BroadcastHelper& helper) {
          const auto sum = helper.ScalarInput0<float>() + helper.EigenInput1<float>().array();
          if (Relu) helper.OutputEigen<float>() = sum.cwiseMax(0.0f);
          else helper.OutputEigen<float>() = sum;
        },
        [](BroadcastHelper& helper) {
          const auto sum = helper.EigenInput0<float>().array() + helper.ScalarInput1<float>();
          if (Relu) helper.OutputEigen<float>() = sum.cwiseMax(0.0f);
          else helper.OutputEigen<float>() = sum;
        },
        [](BroadcastHelper& helper) {
          const auto sum = helper.EigenInput0<float>() + helper.EigenInput1<float>();
          if (Relu) helper.OutputEigen<float>() = sum.cwiseMax(0.0f);
          else helper.OutputEigen<float>() = sum;
        }};
    UntypedBroadcastTwo(*context, funcs, Relu ? 2.0 : 1.0);
    return Status::OK();
  }

 private:
  char mode_;
};

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    Add, kOnnxDomain, 7, 12, float, kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), FloatAdd<false>);
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    Add, kOnnxDomain, 13, 13, float, kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), FloatAdd<false>);
ONNX_OPERATOR_TYPED_KERNEL_EX(
    Add, kOnnxDomain, 14, float, kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), FloatAdd<false>);
ONNX_OPERATOR_KERNEL_EX(
    AddRelu, kOnnxDomain, 1, kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), FloatAdd<true>);

}  // namespace systolic
}  // namespace onnxruntime
#endif  // SYSTOLIC_FP32
