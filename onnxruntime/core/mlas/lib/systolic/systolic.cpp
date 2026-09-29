#ifdef SYSTOLIC_FP16
// FP16 implementation shared by ORT and standalone replay binaries.
#include "core/mlas/inc/systolic_mlas.h"
#include "core/mlas/inc/systolic_fp16.h"
#include "core/common/replay_profile.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "conv_rect.h"

#ifdef __riscv
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "systolic_include.h"
#pragma GCC diagnostic pop
static_assert(DIM == 32 && sizeof(elem_t) == 2 && sizeof(acc_t) == 4,
              "FP16 kernels require DPVO FP16 DIM32 ABI");
#endif

namespace {
uint16_t HalfFma(uint16_t a, uint16_t b, uint16_t c) {
  // Preserve the numerical reference: fused double arithmetic followed by a
  // direct double-to-half conversion, never an intermediate float rounding.
  const double value = std::fma(static_cast<double>(systolic_fp16::FromBits(a)),
                                static_cast<double>(systolic_fp16::FromBits(b)),
                                static_cast<double>(systolic_fp16::FromBits(c)));
  return systolic_fp16::ToBits(static_cast<_Float16>(value));
}

void CheckHalfMode(char mode) {
  systolic_fp16::RequireRoundToNearest();
  if (mode != 0 && mode != 2)
    throw std::invalid_argument("FP16 Gemmini supports CPU mode 0 or WS mode 2 only");
#ifndef __riscv
  if (mode != 0) throw std::invalid_argument("Gemmini WS requires RISC-V hardware");
#endif
}
}

void SystolicHalfMatmul(char mode, size_t m, size_t n, size_t k,
                       const uint16_t* a, size_t lda,
                       const uint16_t* b, size_t ldb,
                       const float* d, size_t ldd,
                       uint16_t* c, size_t ldc,
                       bool repeating_bias, bool relu, float output_scale,
                       float* full_output) {
  CheckHalfMode(mode);
  if (m == 0 || n == 0) return;
  if ((!c && !full_output) || ldc < n || (k && (!a || !b || lda < k || ldb < n)) || (d && ldd < n))
    throw std::invalid_argument("Invalid FP16 matmul buffers/strides");
  if (full_output && (relu || output_scale != 1.0f))
    throw std::invalid_argument("Full accumulator read does not apply activation/scaling");
  ort_replay::Scope profile("kernel", "matmul.fp16", "MatMul", "SystolicExecutionProvider");
  if (profile.Active()) {
    profile.Detail(("dtype=fp16;pe_acc=fp16;acc=fp32;DIM=32;execution=" +
        std::to_string(int(mode)) + ";M=" + std::to_string(m) +
        ";N=" + std::to_string(n) + ";K=" + std::to_string(k)).c_str());
  }
#ifdef __riscv
  if (mode == 2 && k) {
    tiled_matmul_auto(m, n, k, a, b, d, full_output ? static_cast<void*>(full_output) : c, lda, ldb, ldd, ldc,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        relu ? RELU : NO_ACTIVATION, output_scale, 0, repeating_bias,
        false, false, full_output != nullptr, false, 3, WS);
    return;
  }
#endif
  // Numerical reference: half FMA in groups of DIM, float accumulator between
  // groups. RTL reduction ordering still needs independent hardware validation.
  for (size_t i = 0; i < m; ++i) for (size_t j = 0; j < n; ++j) {
    float sum = d ? d[(repeating_bias ? 0 : i) * ldd + j] : 0.0f;
    for (size_t kk = 0; kk < k; kk += 32) {
      uint16_t partial = 0;
      for (size_t t = kk; t < std::min(k, kk + 32); ++t)
        partial = HalfFma(a[i * lda + t], b[t * ldb + j], partial);
      sum += static_cast<float>(systolic_fp16::FromBits(partial));
    }
    sum *= output_scale;
    if (relu && sum < 0) sum = 0;
    if (full_output) full_output[i * ldc + j] = sum;
    else c[i * ldc + j] = systolic_fp16::ToBits(static_cast<_Float16>(sum));
  }
}

bool SystolicHalfConvRect(char mode, int64_t batch,
                         int64_t ih, int64_t iw, int64_t ci,
                         int64_t co, int64_t oh, int64_t ow,
                         int64_t stride, int64_t pad, int64_t kernel,
                         const uint16_t* input, const uint16_t* weights,
                         const float* bias, uint16_t* output, bool relu) {
  CheckHalfMode(mode);
  const systolic_rect::Shape s{batch, ih, iw, ci, co, oh, ow, kernel, stride, pad};
  if (mode != 2 || !input || !weights || !output || !systolic_rect::Supported(s)) return false;
#if defined(__riscv) && defined(SYSTOLIC_FP16_RECT_SCALE_VERIFIED)
  // Current LoopConv.scala hardcodes FP32 mvin identity. Only enable after
  // an independently validated hardware fix and a matching bitstream.
  const auto tile = systolic_rect::SelectTile(s, DIM, BANK_NUM * BANK_ROWS, ACC_ROWS);
  if (!tile.rows) return false;
  ort_replay::Scope profile("kernel", "conv.direct.fp16", "Conv", "SystolicExecutionProvider");
  if (profile.Active()) profile.Detail("dtype=fp16;pe_acc=fp16;acc=fp32;abi=rect_v1;DIM=32");
  gemmini_extended_config_st(co * sizeof(elem_t), relu, ACC_SCALE_IDENTITY);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
      0, 1, stride, false, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
  systolic_rect::Run(s, tile, input, weights, bias, output, relu,
      [](int funct, uint64_t rs1, uint64_t rs2) {
#define HALF_RECT_EMIT(f) case f: ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, f); break
        switch (funct) {
          HALF_RECT_EMIT(16); HALF_RECT_EMIT(17); HALF_RECT_EMIT(18);
          HALF_RECT_EMIT(19); HALF_RECT_EMIT(20); HALF_RECT_EMIT(21); HALF_RECT_EMIT(15);
        }
#undef HALF_RECT_EMIT
      });
  gemmini_fence();
  return true;
#else
  (void)bias; (void)relu;
  return false;
#endif
}

void SystolicFlush() {
#ifdef __riscv
  gemmini_flush(0);
#endif
}
#else
// See LICENSE for license details.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdexcept>
#include "core/common/replay_profile.h"
#ifdef SYSTOLIC_FP32
#include "core/mlas/inc/mlas.h"
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#include "systolic_include.h"
#pragma GCC diagnostic pop

#ifdef SYSTOLIC_FP32
#include "conv_rect.h"

// NHWC/HWIO rectangular WS convolution; false means no instructions issued.
bool SystolicConvRect(char accelerator_mode, int64_t batch,
                     int64_t input_h, int64_t input_w, int64_t input_channels,
                     int64_t output_channels, int64_t output_h, int64_t output_w,
                     int64_t stride, int64_t padding, int64_t kernel,
                     const float* input, const float* weights, const float* bias,
                     float* output, bool relu, float output_scale) {
  const systolic_rect::Shape s{batch, input_h, input_w, input_channels,
      output_channels, output_h, output_w, kernel, stride, padding};
  if (accelerator_mode != 2 || !input || !weights || !output ||
      !systolic_rect::Supported(s)) return false;
  const auto tile = systolic_rect::SelectTile(s, DIM, BANK_NUM * BANK_ROWS, ACC_ROWS);
  if (!tile.rows) return false;
  ort_replay::Scope profile("kernel", "conv.direct", "Conv", "SystolicExecutionProvider");
  if (profile.Active()) {
    const std::string detail = "path=direct_conv;layout=NHWC;abi=rect_v1;tiling=capacity_v2;tile=" +
        std::to_string(tile.rows) + "x" + std::to_string(tile.cols) + "x" +
        std::to_string(tile.ci) + "x" + std::to_string(tile.co) +
        ";loops=" + std::to_string(static_cast<uint64_t>(systolic_rect::LoopCount(s, tile)));
    profile.Detail(detail.c_str());
  }
  gemmini_extended_config_st(output_channels * sizeof(float), relu, output_scale);
  // Explicitly disable the separate HW Im2Col unit, even after a prior user.
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
      0, 1, stride, false, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
  systolic_rect::Run(s, tile, input, weights, bias, output, relu,
      [](int funct, uint64_t rs1, uint64_t rs2) {
        // RoCC funct is an assembler immediate; retain literal cases.
#define RECT_EMIT(f) case f: ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, f); break
        switch (funct) {
          RECT_EMIT(16); RECT_EMIT(17); RECT_EMIT(18); RECT_EMIT(19);
          RECT_EMIT(20); RECT_EMIT(21); RECT_EMIT(15);
        }
#undef RECT_EMIT
      });
  gemmini_fence();
  return true;
}
#endif


/**
 * Perform a matmul and subsequent quantization.
 * Switch between TILED_OS and TILED_CPU
 * 
 * Elements are accumulated internally into acc_t (int32) and subsequently rounded/saturated to elem_t (int8).
 * The given divisor *must* be a power of 2.
 */

#define ROTATED_MATMUL_TYPE(x)

/**
 * Interally CPU is last in tiled_matmul_type_t but we want to expose CPU as accelerator mode 0
 * So just rotate everything by one
 */
inline int positive_mod(int i, int n) {
  return (i % n + n) % n;
}
inline tiled_matmul_type_t get_accelerator_mode(int mode) {
  return static_cast<tiled_matmul_type_t>(positive_mod(mode - 1, (int)CPU + 1));
}

/* Internal -- no need to touch */


/**
 * Wrapper function around tiled_matmul_auto that provides a BLAS like interface
 * C := alpha*op( A )op( B ) + beta*D
 * Note that like blas, dim_I dim_J and dim_K are after the transpose is applied
 * 
 * No output scale is applied, so this is best used with floating point types
 */
void tiled_gemm_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                     size_t strideA,
                     size_t strideB,
                     size_t strideD,
                     size_t strideC,
                     const elem_t* A, const elem_t* B,
                     const acc_t* D, elem_t* C,
                     int act, scale_t scaleAlpha, acc_scale_t scaleBeta, bool repeating_bias,
                     bool transA, bool transB,
                     enum tiled_matmul_type_t tiled_matmul_type) {
  tiled_matmul_auto(dim_I, dim_J, dim_K,
                    A, B, D, C,
                    strideA, strideB, strideD, strideC,
                    scaleAlpha, MVIN_SCALE_IDENTITY, scaleBeta,
                    act, ACC_SCALE_IDENTITY, /*relu6_shift= */ 0, repeating_bias,
                    transA, transB,
                    /*full_c= */ false, /*low_d= */ false,
                    /*weightA= */ 3,
                    tiled_matmul_type);
}

/**
 * Wrapper function around above tiled_gemm_auto that assumes full stride (equal to matrix width)
 */
void tiled_gemm_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                     const elem_t* A, const elem_t* B,
                     const acc_t* D, elem_t* C,
                     int act, scale_t scaleAlpha, acc_scale_t scaleBeta, bool repeating_bias,
                     bool transA, bool transB,
                     enum tiled_matmul_type_t tiled_matmul_type) {
  int lda = transA ? dim_I : dim_K;
  int ldb = transB ? dim_K : dim_J;
  tiled_gemm_auto(dim_I, dim_J, dim_K, lda, ldb, dim_J, dim_J,
                  A, B, D, C,
                  act, scaleAlpha, scaleBeta, repeating_bias,
                  transA, transB, tiled_matmul_type);
}

/**
 * Wrapper function around tiled_matmul_auto that provides a simple interface to
 * call for matrix/matrix multiplication C = scale*(A*B + D)
 */
void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                       size_t strideA,
                       size_t strideB,
                       size_t strideD,
                       size_t strideC,
                       const elem_t* A, const elem_t* B,
                       const acc_t* D, elem_t* C,
                       int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
                       bool transA, bool transB,
                       enum tiled_matmul_type_t tiled_matmul_type) {
  tiled_matmul_auto(dim_I, dim_J, dim_K,
                    A, B, D, C,
                    strideA, strideB, strideD, strideC,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
                    act, scale, relu6_shift, repeating_bias,
                    transA, transB,
                    /*full_c= */ false, /*low_d= */ false,
                    /*weightA= */ 3,
                    tiled_matmul_type);
}

/**
 * Wrapper function around above that assumes stride is full matrix width
 */
void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                       const elem_t* A, const elem_t* B,
                       const acc_t* D, elem_t* C,
                       int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
                       enum tiled_matmul_type_t tiled_matmul_type) {
  tiled_matmul_auto(dim_I, dim_J, dim_K, dim_K, dim_J, dim_J, dim_J,
                    A, B, D, C,
                    act, scale, relu6_shift, repeating_bias,
                    /*transA= */ false, /*transB= */ false, tiled_matmul_type);
}

/* End internal */

// Logical dimensions and DIM padding describe work shape, not measured PE utilization.
static void ProfileMatmulShape(ort_replay::Scope& profile, size_t m, size_t n,
                               size_t k, char mode, bool trans_a, bool trans_b) {
  if (!profile.Active()) return;
  const auto pad = [](size_t v) { return ((v + DIM - 1) / DIM) * DIM; };
  const std::string detail = "M=" + std::to_string(m) + ";N=" + std::to_string(n) +
      ";K=" + std::to_string(k) + ";execution=" + std::to_string(int(mode)) +
      ";DIM=" + std::to_string(DIM) + ";padded_M=" + std::to_string(pad(m)) +
      ";padded_N=" + std::to_string(pad(n)) + ";padded_K=" + std::to_string(pad(k)) +
      ";transA=" + std::to_string(trans_a) + ";transB=" + std::to_string(trans_b) +
      ";tiling=auto;ops=" + std::to_string(2.0 * m * n * k);
  profile.Detail(detail.c_str());
}



/**
 * An interface similar to Gemmlowp's matrix multiply
 * Does real_multiplier*(in1 * in2 + bias)
 */
void SystolicMultiply(char accelerator_mode, bool relu, int dimI, int dimJ, int dimK,
                      const elem_t* in1, const elem_t* in2, elem_t* out, acc_scale_t real_multiplier, const acc_t* bias) {
  ort_replay::Scope profile("kernel", "matmul", "MatMul", "SystolicExecutionProvider");
  ProfileMatmulShape(profile, dimI, dimJ, dimK, accelerator_mode, false, false);
#ifndef FOR_FIRESIM
  if (!ort_replay::Enabled("total")) printf("Called into systolic matmul!\n");
  if (!ort_replay::Enabled("total")) printf("Using accelerated matmul with dimensions (%d, %d, %d)\n", dimI, dimJ, dimK);
#endif
  tiled_matmul_auto(dimI, dimJ, dimK, in1, in2, bias, out, /*activation= */ relu,
                    real_multiplier,
                    /*relu6_shift= */ 0, /* repeating_bias= */ 0,
                    get_accelerator_mode(accelerator_mode));
}

#ifdef SYSTOLIC_FP32
/**
 * Provides an interface similar to BLAS matrix multiply
 * C = alpha*A*B + beta*C
 */
void SystolicGemm(char accelerator_mode,
                  bool TransA,
                  bool TransB,
                  size_t M,
                  size_t N,
                  size_t K,
                  scale_t alpha,
                  const elem_t* A,
                  const elem_t* B,
                  acc_scale_t beta,
                  elem_t* C) {
  ort_replay::Scope profile("kernel", "gemm", "Gemm", "SystolicExecutionProvider");
  ProfileMatmulShape(profile, M, N, K, accelerator_mode, TransA, TransB);
#ifndef FOR_FIRESIM
  if (!ort_replay::Enabled("total")) printf("Called into systolic gemm!\n");
  if (!ort_replay::Enabled("total")) printf("Using accelerated gemm with dimensions (%zd, %zd, %zd)\n", M, N, K);
#endif

  tiled_gemm_auto(M, N, K, A, B, beta == 0 ? nullptr : C, C, /*activation= */ false,
                  alpha, beta, /* repeating_bias= */ 0,
                  TransA, TransB, get_accelerator_mode(accelerator_mode));
}

void SystolicGemm(char accelerator_mode,
                  bool TransA,
                  bool TransB,
                  size_t M,
                  size_t N,
                  size_t K,
                  scale_t alpha,
                  const elem_t* A,
                  int lda,
                  const elem_t* B,
                  int ldb,
                  acc_scale_t beta,
                  elem_t* C,
                  int ldc) {
  ort_replay::Scope profile("kernel", "gemm", "Gemm", "SystolicExecutionProvider");
  ProfileMatmulShape(profile, M, N, K, accelerator_mode, TransA, TransB);
#ifndef FOR_FIRESIM
  if (!ort_replay::Enabled("total")) printf("Called into systolic gemm!\n");
  if (!ort_replay::Enabled("total")) printf("Using accelerated gemm with dimensions (%zd, %zd, %zd)\n", M, N, K);
#endif
  tiled_gemm_auto(M, N, K,
                 lda, ldb, ldc, ldc,
                 A, B, beta == 0 ? nullptr : C, C, /*activation= */ false,
                  alpha, beta, /* repeating_bias= */ 0,
                  TransA, TransB, get_accelerator_mode(accelerator_mode));
}
#endif

/**
 * Provides a matrix multiply that allows specifying strides
 */
void SystolicMultiply(char accelerator_mode, bool relu,
                      int dimI, int dimJ, int dimK,
                      const elem_t* in1, int strideIn1,
                      const elem_t* in2, int strideIn2,
                      elem_t* out, int strideOut,
                      acc_scale_t real_multiplier,
                      const acc_t* bias, int strideBias, bool repeating_bias) {
  ort_replay::Scope profile("kernel", "matmul", "MatMul", "SystolicExecutionProvider");
  ProfileMatmulShape(profile, dimI, dimJ, dimK, accelerator_mode, false, false);
#ifndef FOR_FIRESIM
  if (!ort_replay::Enabled("total")) printf("Called into systolic matmul!\n");
  if (!ort_replay::Enabled("total")) printf("Using accelerated matmul with dimensions (%d, %d, %d)\n", dimI, dimJ, dimK);
#endif
  tiled_matmul_auto(dimI, dimJ, dimK,
                    strideIn1, strideIn2, strideBias, strideOut,
                    in1, in2, bias, out, /*activation= */ relu,
                    real_multiplier, /*relu6_shift= */ 0, /* repeating_bias= */ repeating_bias,
                    /*transA= */ false, /*transB= */ false,
                    get_accelerator_mode(accelerator_mode));
}

/**
 * Adds two matrices elementwise
 */
void SystolicAdd(char accelerator_mode __attribute__((unused)), bool relu, const elem_t* A, float A_scale, const elem_t* B,
                 float B_scale,
                 elem_t* C, float C_scale, int dim) {
#ifndef FOR_FIRESIM
  if (!ort_replay::Enabled("total")) printf("Called into systolic add\n");
#endif
  // To most efficiently use systolic, instead of using 1xdim, we use 16xResizedDim.
  // Systolic can load multiple blocks in a given row

  // Note that it's more accurate to use A_scale/C_scale and B_scale/C_scale as the A, B scales (with C_scale = 1)
  // Since that way we don't blow up rounding error by dividing by C_scale

  // Equivalent to:
  // for (int i = 0; i < dim; i++) {
  //   int32_t tmp1 = (int) MVIN_SCALE(*A, A_scale/C_scale);
  //   int32_t tmp2 = (int) MVIN_SCALE(*B, B_scale/C_scale);
  //   *C = scale_and_sat(tmp1 + tmp2, relu ? RELU : 0, 1, 0);

  //   A++;
  //   B++;
  //   C++;
  // }

  int resizedDim = dim - dim % DIM;
  tiled_resadd_auto(DIM, resizedDim / DIM, A_scale / C_scale, B_scale / C_scale,
                    /*C_scale= */ 1, A, B, C, relu, get_accelerator_mode(accelerator_mode));
  if (dim % DIM > 0) {
#ifndef FOR_FIRESIM
    if (!ort_replay::Enabled("total")) printf("Some extra leftover\n");
#endif
    tiled_resadd_auto(1, dim % DIM, A_scale / C_scale, B_scale / C_scale,
                      /*C_scale= */ 1, A + resizedDim, B + resizedDim, C + resizedDim, relu, get_accelerator_mode(accelerator_mode));
  }
}

/**
 * Convolution of two matrices. Input must be in NHWC format, weight must be in HWIO format
 */
void SystolicConv(char accelerator_mode, int batch_size, int in_dim, int in_channels,
                  int out_channels, int out_dim,
                  int stride, int padding, int kernel_dim,
                  const elem_t* input,
                  const elem_t* weights,
                  const acc_t* bias,
                  elem_t* output,
                  bool relu,
                  float output_scale,
                  int pool_size, int pool_stride, int pool_padding) {
  if (!ort_replay::Enabled("total")) printf("Called into systolic conv\n");
  if (pool_size != 0) {
    if (!ort_replay::Enabled("total")) printf("Using systolic pooling\n");
  }
  // printf("Debugging info\n");
  // printf("Batch size, in_w/h, in_channel %d %d %d\n", batch_size, in_dim, in_channels);
  // printf("Out_channels, out_w/h %d %d\n", out_channels, out_dim);
  // printf("Stride, padding %d %d\n", stride, padding);
  // printf("kernel_w/h %d\n", kernel_dim);
  // if (bias) {
  //   printf("Bias values: %d\n", bias[0]);
  // }
  // printf("Relu? %d\n", relu);

  tiled_conv_auto(batch_size, in_dim, in_channels, out_channels, out_dim,
                  stride,
                  /*input_dilation= */ 1,
                  /*kernel_dilation= */ 1,
                  padding, kernel_dim,
                  /*wrot180= */ false, 
                  /*trans_output_1203= */ false,
                  /*trans_input_3120= */ false,
                  /*trans_weight_1203= */ false,
                  /*trans_weight_0132= */ false,
                  input, weights, bias, output,
                  relu, output_scale, /*relu6_shift= */ 0,
                  pool_size, pool_stride, pool_padding,
                  get_accelerator_mode(accelerator_mode));

  // printf("Output\n");
  // for (int i = 0; i < out_dim * out_dim * out_channels * batch_size; i++) {
  //   printf("%d ", output[i]);
  // }
  // printf("\n");
}

void SystolicConvTranspose(char accelerator_mode, int batch_size, int in_dim, int in_channels,
                  int out_channels, int out_dim,
                  int stride, int padding, int kernel_dim,
                  const elem_t* input,
                  const elem_t* weights,
                  const acc_t* bias,
                  elem_t* output,
                  bool relu,
                  float output_scale) {
  if (!ort_replay::Enabled("total")) printf("Called into systolic conv transpose\n");


  tiled_conv_auto(batch_size, in_dim, in_channels, out_channels, out_dim,
                  /*stride = */ 1,
                  /*input_dilation= */ stride,
                  /*kernel_dilation= */ 1,
                  /*padding= */ kernel_dim - 1 - padding,
                  kernel_dim,
                  /*wrot180= */ true,
                  /*trans_output_1203= */ false,
                  /*trans_input_3120= */ false,
                  /*trans_weight_1203= */ false,
                  /*trans_weight_0132= */ true,
                  input, weights, bias, output,
                  relu, output_scale, /*relu6_shift= */ 0,
                  0, 0, 0,
                  get_accelerator_mode(accelerator_mode));
}


/**
 * Note that the batch size and dimensions are _after_ transposition is applied
 */
void SystolicConvBackpropFilter(char accelerator_mode, int batch_size, int in_dim, int in_channels,
                  int out_channels, int out_dim,
                  int stride, int padding, int kernel_dim,
                  const elem_t* input,
                  const elem_t* weights,
                  const acc_t* bias,
                  elem_t* output,
                  bool relu,
                  float output_scale) {
  if (!ort_replay::Enabled("total")) printf("Called into systolic conv backprop filter\n");


  tiled_conv_auto(batch_size, in_dim, in_channels, out_channels, out_dim,
                  /*stride = */ 1,
                  /*input_dilation= */ 1,
                  /*kernel_dilation= */ stride,
                  /*padding= */ padding,
                  kernel_dim,
                  /*wrot180= */ false, 
                  /*trans_output_1203= */ true,
                  /*trans_input_3120= */ true,
                  /*trans_weight_1203= */ true,
                  /*trans_weight_0132= */ false,
                  input, weights, bias, output,
                  relu, output_scale, /*relu6_shift= */ 0,
                  0, 0, 0,
                  get_accelerator_mode(accelerator_mode));
}

void SystolicFlush() {
  // FLUSH does not wait for queued Gemmini commands, so drain them first.
  gemmini_fence();
  gemmini_flush(0);
}

// We do this to clear out gemmini on every process launch
#ifdef FOR_FIRESIM
__attribute__((constructor))
void cleargemmini() {
  SystolicFlush();
}
#endif

#endif  // SYSTOLIC_FP16

// The CGR1 command sequence is provided by the same compiled implementation as
// the other Systolic wrappers. DPVO_CORR_GATHER selects the runner call site;
// shared MLAS builds do not carry that application flag. Defining these symbols
// issues no CGR1 instructions. Call only after explicit matched-build opt-in.
#if defined(__riscv) && defined(SYSTOLIC_FP32) && !defined(SYSTOLIC_FP16)
static_assert(DIM == 16 && BANK_NUM == 4 && BANK_ROWS == 1024 && ACC_ROWS == 1024,
              "Correlation gather requires the FP32 DIM16 256KiB-SPAD / 64KiB-ACC ABI");
static_assert(sizeof(elem_t) == 4 && sizeof(acc_t) == 4, "FP32 gather ABI");

namespace systolic_corr {
// The legacy fence macro lacks a compiler memory clobber. The direct-SPAD
// wrapper must also order CPU feature stores and prevent cached C loads.
static inline void Fence() { asm volatile("fence" ::: "memory"); }

void Probe() {
  Fence();
  const uint64_t status = gemmini_corr_status(GEMMINI_CORR_COUNTER_STATUS);
  if ((status & GEMMINI_CORR_MAGIC_MASK) != GEMMINI_CORR_MAGIC ||
      !(status & GEMMINI_CORR_ENABLED) || (status & (GEMMINI_CORR_BUSY | GEMMINI_CORR_ERROR)))
    throw std::runtime_error("--corr-gather requires matching enabled CGR1 hardware; invalid status");
}

static void LoadA(const float* a) {
  // Do not depend on load state left by a convolution or another matmul.
  Fence();
  gemmini_extended3_config_ld(128 * sizeof(float), MVIN_SCALE_IDENTITY, false, 0);
  asm volatile("" ::: "memory");
  for (uint32_t k = 0; k < 8; ++k) {
    gemmini_extended_mvin(a + 16 * k, GEMMINI_CORR_A_BASE + 16 * k, 16, 1);
  }
  Fence();
}

static uint64_t WaitGather() {
  Fence();
  const uint64_t status = gemmini_corr_status(GEMMINI_CORR_COUNTER_STATUS);
  if ((status & GEMMINI_CORR_MAGIC_MASK) != GEMMINI_CORR_MAGIC ||
      !(status & GEMMINI_CORR_ENABLED) || (status & (GEMMINI_CORR_BUSY | GEMMINI_CORR_ERROR)))
    throw std::runtime_error("Correlation gather failed or incomplete; refusing to consume SPAD B");
  const uint64_t rows = gemmini_corr_status(GEMMINI_CORR_COUNTER_ROWS);
  if (rows != GEMMINI_CORR_B_ROWS)
    throw std::runtime_error("Correlation gather row count mismatch; refusing to consume SPAD B");
  return rows;
}

GatherStatistics Gather(const float* a, const Window& level0, const Window& level1) {
  LoadA(a);
  gemmini_corr_config(reinterpret_cast<uint64_t>(level0.data),
      reinterpret_cast<uint64_t>(level1.data), level0.height, level0.width,
      level1.height, level1.width);
  asm volatile("" ::: "memory");
  gemmini_corr_start(level0.x, level0.y, level1.x, level1.y, GEMMINI_CORR_B_BASE);
  const uint64_t rows = WaitGather();
  return {gemmini_corr_status(GEMMINI_CORR_COUNTER_CYCLES),
      gemmini_corr_status(GEMMINI_CORR_COUNTER_USEFUL_BYTES),
      gemmini_corr_status(GEMMINI_CORR_COUNTER_STRIPS), rows,
      gemmini_corr_status(GEMMINI_CORR_COUNTER_READ_STALL),
      gemmini_corr_status(GEMMINI_CORR_COUNTER_WRITE_STALL)};
}

void DotPreloaded(float* output) {
  gemmini_extended_config_st(128 * sizeof(float), NO_ACTIVATION, ACC_SCALE_IDENTITY);
  // Explicitly reset A/C strides, transpose, activation and Im2Col configuration.
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY,
      0, 1, 1, false, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, UINT64_C(15),
      (UINT64_C(8) << 32) | (UINT64_C(8) << 16) | 1, k_LOOP_WS_CONFIG_BOUNDS);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, UINT64_C(0), UINT64_C(0), k_LOOP_WS_CONFIG_ADDRS_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, UINT64_C(0), (uint64_t)output, k_LOOP_WS_CONFIG_ADDRS_DC);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, UINT64_C(128), UINT64_C(128), k_LOOP_WS_CONFIG_STRIDES_AB);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, UINT64_C(128), UINT64_C(128), k_LOOP_WS_CONFIG_STRIDES_DC);
  // funct24: rs2 is B's exclusive END, not its base.
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)GEMMINI_CORR_A_BASE,
      (uint64_t)(GEMMINI_CORR_B_BASE + GEMMINI_CORR_B_ROWS), 24);
  // rs1=0: explicit A/B SPAD ids, no bias/activation/initial accumulation.
  // Skip A/B only. D=0 still runs empty ldD to advance its accumulator cursor.
  // spad_only=0 leaves normal accumulator-to-memory output enabled.
  // Restore this loop slot's original SPAD bounds before ordinary MLAS reuse.
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, UINT64_C(0), (UINT64_C(1) << 3) | (UINT64_C(1) << 4) |
      GEMMINI_CORR_RESTORE_SPAD_DEFAULTS, k_LOOP_WS);
  Fence();
}
}  // namespace systolic_corr
#endif
