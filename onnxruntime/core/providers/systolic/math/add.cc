// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef SYSTOLIC_FP32
#include "core/common/replay_profile.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/systolic_mlas.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/providers/systolic/systolic_fwd.h"

#include <cstdlib>
#include <string>

namespace onnxruntime {
namespace systolic {

// Row bias: `bias` is [N] or [1,...,1,N] and `x` is contiguous [...,N], so
// the broadcast output has x's shape and every row of x adds the same bias.
inline bool RowBiasColumns(const TensorShape& bias, const TensorShape& x, size_t& cols) {
  const size_t bias_rank = bias.NumDimensions(), x_rank = x.NumDimensions();
  if (bias_rank == 0 || x_rank < bias_rank) return false;
  for (size_t d = 0; d + 1 < bias_rank; ++d) {
    if (bias[d] != 1) return false;
  }
  const int64_t n = bias[bias_rank - 1];
  if (n <= 0 || x[x_rank - 1] != n) return false;
  cols = static_cast<size_t>(n);
  return true;
}

// Narrow rows (for example the 2-wide update heads) would issue one short DMA
// row per mvin; they stay on the CPU broadcast path.
constexpr size_t kMinBiasResaddColumns = 16;

// ORT_SYSTOLIC_BIAS_RESADD=0 keeps every broadcast Add on the CPU path (for
// isolating this kernel in end-to-end measurements). Read once per process.
inline bool BiasResaddEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("ORT_SYSTOLIC_BIAS_RESADD");
    return value == nullptr || std::string(value) != "0";
  }();
  return enabled;
}

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

    // Row-bias broadcast ([N] + [...,N]): stride-0 bias resadd on Gemmini.
    // fl(x + b) in the accumulator equals the CPU sum bit for bit.
    size_t cols = 0;
    const Tensor* x = nullptr;
    const Tensor* bias = nullptr;
    if (RowBiasColumns(b->Shape(), a->Shape(), cols)) {
      x = a; bias = b;
    } else if (RowBiasColumns(a->Shape(), b->Shape(), cols)) {
      x = b; bias = a;
    }
    if (x != nullptr && cols >= kMinBiasResaddColumns && BiasResaddEnabled()) {
      auto* output = context->Output(0, x->Shape());
      const auto count = static_cast<size_t>(x->Shape().Size());
      if (count == 0) return Status::OK();
      if (SystolicBiasAdd(mode_, Relu, x->Data<float>(), bias->Data<float>(),
                          output->MutableData<float>(), count / cols, cols)) {
        return Status::OK();
      }
    }

    // Preserve ONNX multidirectional broadcasting without materializing full
    // expanded tensors or issuing tiny accelerator transfers for each span.
    // Allocate the output before the profile scope so the kernel time does not
    // include page population; UntypedBroadcastTwo reuses this allocation.
    context->Output(0, InputBroadcaster(*a, *b).GetOutputShape());
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
