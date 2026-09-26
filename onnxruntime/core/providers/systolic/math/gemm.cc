// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gemm.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/providers/cpu/math/gemm_helper.h"

#ifdef SYSTOLIC_FP16
#include "core/mlas/inc/systolic_mlas.h"
#include "core/mlas/inc/systolic_fp16.h"
#endif

#ifdef SYSTOLIC_FP32

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    7, 13,
    float,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

template <typename T>
static void GemmBroadcastBias(int64_t M, int64_t N, float beta,
                              const T* c_data, const TensorShape* c_shape,
                              T* y_data) {
  // Broadcast the bias as needed if bias is given
  if (beta != 0 && c_data != nullptr) {
    ORT_ENFORCE(c_shape != nullptr, "c_shape is required if c_data is provided");
    auto output_mat = EigenMatrixMapRowMajor<T>(y_data, M, N);
    if (c_shape->Size() == 1) {
      // C is (), (1,) or (1, 1), set the scalar
      output_mat.setConstant(*c_data);
    } else if (c_shape->NumDimensions() == 1 || (*c_shape)[0] == 1) {
      // C is (N,) or (1, N)
      output_mat.rowwise() = ConstEigenVectorMap<T>(c_data, N).transpose();
    } else if ((*c_shape)[1] == 1) {
      // C is (M, 1)
      output_mat.colwise() = ConstEigenVectorMap<T>(c_data, M);
    } else {
      // C is (M, N), no broadcast needed.
      output_mat = ConstEigenMatrixMapRowMajor<T>(c_data, M, N);
    }
  }
}

template <>
Status Gemm<float>::Compute(OpKernelContext* context) const {
  const auto* A = context->Input<Tensor>(0);
  const auto* B = context->Input<Tensor>(1);
  const auto* C = context->Input<Tensor>(2);

  // Bias could be missing. Treat as scalar 0 if that is the case.
  GemmHelper helper(A->Shape(), trans_A_, B->Shape(), trans_B_,
                    C != nullptr ? C->Shape() : TensorShape({}));

  if (!helper.State().IsOK())
    return helper.State();

  int64_t M = helper.M();
  int64_t N = helper.N();
  int64_t K = helper.K();

  //printf("M, N, K: %d %d %d\n", (int) M, (int) N, (int) K);

  auto Y = context->Output(0, {M, N});

  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M == 0 || N == 0)
    return Status::OK();

  float* y_data = Y->MutableData<float>();

  const float* c_data = C != nullptr ? C->Data<float>() : nullptr;
  const TensorShape* c_shape = C != nullptr ? &C->Shape() : nullptr;

  // if input is empty tensor, return directly as nothing need to be calculated.
  if (M == 0 || N == 0)
    return Status::OK();

  // Broadcast the bias as needed if bias is given
  GemmBroadcastBias(M, N, beta_, c_data, c_shape, y_data);

  // printf("A matrix\n");
  // PrintMatrix(M, K, A->Data<float>());
  // printf("B matrix\n");
  // PrintMatrix(K, N, B->Data<float>());
  // printf("Bias matrix\n");
  // PrintMatrix(M, N, y_data);

  char acc_mode = static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode();
  SystolicGemm(acc_mode,
                trans_A_, trans_B_, M, N, K,
                alpha_, A->Data<float>(), B->Data<float>(), c_data != nullptr ? beta_ : 0, y_data);

  // printf("Out matrix\n");
  // PrintMatrix(M, N, y_data);

  // printf("First few output values\n:");
  // for (int i = 0; i < 20; i++) {
  //   printf("%f ", y_data[i]);
  // }
  // printf("\n");

  return Status::OK();
}

}  // namespace systolic
}  // namespace onnxruntime

#endif

#ifdef SYSTOLIC_FP16

namespace onnxruntime {
namespace systolic {

class HalfGemm final : public OpKernel {
 public:
  explicit HalfGemm(const OpKernelInfo& info) : OpKernel(info), mode_(Mode(info)) {
    int64_t ta=0,tb=0;
    info.GetAttrOrDefault("transA", &ta, int64_t(0));
    info.GetAttrOrDefault("transB", &tb, int64_t(0));
    ta_=ta!=0; tb_=tb!=0;
    info.GetAttrOrDefault("alpha", &alpha_, 1.0f);
    info.GetAttrOrDefault("beta", &beta_, 1.0f);
  }
  Status Compute(OpKernelContext* ctx) const override {
    const auto* a=ctx->Input<Tensor>(0); const auto* b=ctx->Input<Tensor>(1);
    const auto* d=ctx->Input<Tensor>(2);
    GemmHelper h(a->Shape(),ta_,b->Shape(),tb_,d ? d->Shape() : TensorShape({}));
    ORT_RETURN_IF_ERROR(h.State());
    const size_t m=h.M(), n=h.N(), k=h.K();
    auto* y=ctx->Output(0,{h.M(),h.N()});
    if (!m || !n) return Status::OK();
    auto av=HalfBits(*a), bv=HalfBits(*b);
    std::vector<uint16_t> ap(m*k),bp(k*n),out(m*n);
    for(size_t i=0;i<m;++i) for(size_t t=0;t<k;++t) ap[i*k+t]=av[ta_ ? t*m+i : i*k+t];
    for(size_t t=0;t<k;++t) for(size_t j=0;j<n;++j) bp[t*n+j]=bv[tb_ ? j*k+t : t*n+j];
    std::vector<float> sums(m*n,0.0f);
    if (alpha_ != 0.0f)
      SystolicHalfMatmul(mode_,m,n,k,ap.data(),k,bp.data(),n,nullptr,n,nullptr,n,false,false,1.0f,sums.data());
    for(size_t i=0;i<m;++i) for(size_t j=0;j<n;++j) {
      float bias=0;
      if (d && beta_ != 0.0f) {
        const auto& ds=d->Shape();
        const size_t row=ds.NumDimensions()==2 && ds[0]!=1 ? i : 0;
        const size_t width=ds.NumDimensions() ? size_t(ds.GetDims().back()) : 1;
        const size_t col=width==1 ? 0 : j;
        bias=static_cast<float>(systolic_fp16::FromBits(d->Data<MLFloat16>()[row*width+col].val));
      }
      out[i*n+j]=systolic_fp16::ToBits(static_cast<_Float16>(alpha_*sums[i*n+j]+beta_*bias));
    }
    SetHalf(*y,out); return Status::OK();
  }
 private: char mode_; bool ta_,tb_; float alpha_,beta_;
};

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(Gemm,kOnnxDomain,7,13,MLFloat16,kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T",DataTypeImpl::GetTensorType<MLFloat16>()),HalfGemm);

}  // namespace systolic
}  // namespace onnxruntime

#endif  // SYSTOLIC_FP16
