// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/providers/systolic/helper/helper.h"

#ifdef SYSTOLIC_FP16
#include "core/mlas/inc/systolic_mlas.h"
#endif

#ifdef SYSTOLIC_FP32

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    9,
    float,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    1, 8,
    float,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);


template <typename T>
Status MatMul<T>::Compute(OpKernelContext* ctx) const {
  const auto* a = ctx->Input<Tensor>(0);
  const auto* b = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  // Using DataRaw as int32_t/uint32_t and int64_t/uint64_t share a common
  // operator body.
  const auto* a_data = reinterpret_cast<const T*>(a->DataRaw());
  const auto* b_data = reinterpret_cast<const T*>(b->DataRaw());
  auto* y_data = reinterpret_cast<T*>(y->MutableDataRaw());

  // TODO: replace it with GemmBatch for performance, it's OK for now as GemmBatch unrolls as well
  size_t max_len = helper.OutputOffsets().size();
  for (size_t i = 0; i < max_len; i++) {
    SystolicMultiply(static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode(),
                            /*relu= */ false,
                            static_cast<int>(helper.M()),
                            static_cast<int>(helper.N()),
                            static_cast<int>(helper.K()),
                            a_data + helper.LeftOffsets()[i],
                            b_data + helper.RightOffsets()[i],
                            y_data + helper.OutputOffsets()[i],
                            /*real_multiplier= */ 1);
  }

  return Status::OK();
}

} // namespace systolic
}  // namespace onnxruntime

#endif

#ifdef SYSTOLIC_FP16

namespace onnxruntime {
namespace systolic {

class HalfMatMul final : public OpKernel {
 public:
  explicit HalfMatMul(const OpKernelInfo& info) : OpKernel(info), mode_(Mode(info)) {}
  Status Compute(OpKernelContext* ctx) const override {
    const auto* a = ctx->Input<Tensor>(0); const auto* b = ctx->Input<Tensor>(1);
    MatMulComputeHelper h;
    ORT_RETURN_IF_ERROR(h.Compute(a->Shape(), b->Shape()));
    auto* y = ctx->Output(0, h.OutputShape());
    if (!y->Shape().Size()) return Status::OK();
    auto av = HalfBits(*a), bv = HalfBits(*b);
    std::vector<uint16_t> out(static_cast<size_t>(y->Shape().Size()));
    for (size_t i=0; i<h.OutputOffsets().size(); ++i)
      SystolicHalfMatmul(mode_, h.M(), h.N(), h.K(),
          av.empty() ? nullptr : av.data()+h.LeftOffsets()[i], h.K(),
          bv.empty() ? nullptr : bv.data()+h.RightOffsets()[i], h.N(),
          nullptr, h.N(), out.data()+h.OutputOffsets()[i], h.N());
    SetHalf(*y, out);
    return Status::OK();
  }
 private: char mode_;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(MatMul,kOnnxDomain,9,MLFloat16,kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T",DataTypeImpl::GetTensorType<MLFloat16>()),HalfMatMul);
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(MatMul,kOnnxDomain,1,8,MLFloat16,kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T",DataTypeImpl::GetTensorType<MLFloat16>()),HalfMatMul);

}  // namespace systolic
}  // namespace onnxruntime

#endif  // SYSTOLIC_FP16
