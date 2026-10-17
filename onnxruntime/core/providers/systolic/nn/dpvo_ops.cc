// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef SYSTOLIC_FP32
#include "core/providers/cpu/nn/dpvo_ops.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/systolic_mlas.h"
#include "core/providers/systolic/gemmini_executor.h"
#include <vector>

namespace onnxruntime {
namespace systolic {

class SoftAgg final : public OpKernel {
 public:
  explicit SoftAgg(const OpKernelInfo& info) : OpKernel(info),
      mode_(static_cast<const SystolicExecutionProvider*>(info.GetExecutionProvider())->GetAcceleratorMode()) {}

  Status PrePack(const Tensor& tensor, int input_index, bool& is_packed) override {
    // Hardware consumes the original weights. CPU fallback must still match
    // the numerical K blocking selected by CPU MatMul's constant-weight pack.
    is_packed = false;
    if (input_index >= 2 && input_index <= 6 && input_index % 2 == 0 &&
        tensor.Shape().NumDimensions() == 2)
      prepack_weights_[static_cast<size_t>((input_index - 2) / 2)] = true;
    return Status::OK();
  }

  Status Compute(OpKernelContext* context) const override {
    dpvo::SoftAggInputs in;
    ORT_RETURN_IF_ERROR(dpvo::ReadSoftAggInputs(context, in));
    if (!in.edges) return Status::OK();
#if defined(__riscv) && !defined(SYSTOLIC_FP16)
    if (mode_ == 2 && in.channels == 384 && in.max_group <= 128) {
      // No catch-and-fallback here: after submit, a failed/partial hardware job
      // is an error, not permission to consume or replay its output.
      SystolicSoftAggResidual(in.x, in.q, in.batches, in.edges, in.channels, in.groups,
          in.parameters[0], in.parameters[1], in.parameters[2], in.parameters[3],
          in.parameters[4], in.parameters[5], in.output);
      return Status::OK();
    }
#endif
    dpvo::SoftAggCpu(in, context->GetOperatorThreadPool(), prepack_weights_);
    return Status::OK();
  }
 private:
  char mode_;
  std::array<bool, 3> prepack_weights_{};
};

class CorrelationBatch final : public OpKernel {
 public:
  explicit CorrelationBatch(const OpKernelInfo& info) : OpKernel(info),
      mode_(static_cast<const SystolicExecutionProvider*>(info.GetExecutionProvider())->GetAcceleratorMode()),
      gather_(info.GetAttrOrDefault<int64_t>("gather", 0) != 0) {}

  Status Compute(OpKernelContext* context) const override {
    dpvo::CorrelationInputs in;
    ORT_RETURN_IF_ERROR(dpvo::ReadCorrelationInputs(context, in));
    if (!in.jobs) return Status::OK();
#if defined(__riscv) && !defined(SYSTOLIC_FP16)
    // The explicit model attribute is a matched-build opt-in. Never probe a
    // legacy bitstream merely to discover whether a custom instruction exists.
    if (mode_ == 2 && gather_ && dpvo::CorrelationGatherSupported(in)) {
      GemminiExecutor::Correlation(in);
      return Status::OK();
    }
    if (mode_ > 0) {
      // Allocate before the first submit. This retains the original CPU-pack
      // + matrix path when only the gather shape/address checks fail.
      std::vector<float> b(in.channels * 128);
      for (size_t j = 0; j < in.jobs; ++j) {
        dpvo::PackCorrelationWindow(in, j, b.data());
        SystolicMultiply(mode_, false, 1, 128, static_cast<int>(in.channels),
            in.a + j * in.channels, b.data(), in.dots + j * 128, 1.0f, nullptr);
      }
      return Status::OK();
    }
#endif
    dpvo::CorrelationCpu(in);
    return Status::OK();
  }
 private:
  char mode_;
  bool gather_;
};

ONNX_OPERATOR_KERNEL_EX(softagg_onchip, kDpvoDomain, 1, kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), SoftAgg);
ONNX_OPERATOR_KERNEL_EX(correlation_batch, kDpvoDomain, 1, kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), CorrelationBatch);
}  // namespace systolic
}  // namespace onnxruntime
#endif  // SYSTOLIC_FP32
