#pragma once
#include <cstddef>
#include <cstdint>
#ifdef SYSTOLIC_FP16
// Raw IEEE binary16 A/B/C; D is FP32. Strides count elements, not bytes.
// mode 0: portable numerical reference; mode 2: DPVO FP16 DIM32 WS hardware.
void SystolicHalfMatmul(char mode, size_t m, size_t n, size_t k,
                       const uint16_t* a, size_t lda,
                       const uint16_t* b, size_t ldb,
                       const float* d, size_t ldd,
                       uint16_t* c, size_t ldc,
                       bool repeating_bias = false, bool relu = false,
                       float output_scale = 1.0f,
                       float* full_output = nullptr);

bool SystolicHalfConvRect(char mode, int64_t batch,
                         int64_t ih, int64_t iw, int64_t ci,
                         int64_t co, int64_t oh, int64_t ow,
                         int64_t stride, int64_t pad, int64_t kernel,
                         const uint16_t* input, const uint16_t* weights,
                         const float* bias, uint16_t* output, bool relu);
#endif
void SystolicFlush();

#ifdef SYSTOLIC_INT8

void SystolicMultiply
MLASCALL(char accelerator_mode, bool relu, int dimI, int dimJ, int dimK, const int8_t* in1, const int8_t* in2,
         int8_t* out, float real_multiplier, const int32_t* bias = nullptr);

void SystolicGemm
MLASCALL(char accelerator_mode, bool relu, int dimI, int dimJ, int dimK, const int8_t* in1, const int8_t* in2,
         int8_t* out, float alpha, float beta, bool transA, bool transB, const int32_t* bias = nullptr);

void SystolicMultiply
MLASCALL(char accelerator_mode, bool relu,
                            int dimI, int dimJ, int dimK,
                            const int8_t* in1, int strideIn1,
                            const int8_t* in2, int strideIn2,
                            int8_t* out, int strideOut,
                            float real_multiplier,
                            const int32_t* bias, int strideBias, bool repeating_bias);

void SystolicAdd
MLASCALL(char accelerator_mode, bool relu, const int8_t* in1, float in1_scale, const int8_t* in2,
         float in2_scale,
         int8_t* out, float out_scale, int dim);

void SystolicConv
MLASCALL(char accelerator_mode, int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int padding, int kernel_dim,
        const int8_t* input,
        const int8_t* weights,
        const int32_t* bias,
        int8_t* output,
        bool relu,
        float output_scale,
        int pool_size = 0, int pool_stride = 0, int pool_padding = 0);

#endif

#ifdef SYSTOLIC_FP32

// Three-pass NHWC InstanceNorm: accelerator reductions/affine, CPU O(C)
// statistics finishing. Requires the opt-in NHWC normalization hardware ABI.
// False leaves output unchanged and issues no accelerator instructions.
// Workspace holds at least 3*channels floats; buffers must not overlap.
bool SystolicInstanceNormNhwc(char accelerator_mode, const float* input,
                             const float* gamma, const float* beta, float* output,
                             size_t batches, size_t spatial, size_t channels,
                             float epsilon, bool relu, float* workspace,
                             size_t workspace_floats);

// Group-1, stride-1, unpadded 1x1 Conv: NHWC input, original OIHW weights,
// and NCHW output. Mode 2 uses WS W * X^T; modes 0/1 are CPU references.
// False means unsupported dimensions/mode/buffers, with no output writes or
// accelerator instructions. Empty batch/spatial dimensions succeed as a no-op.
// Output must not overlap input, weights, or bias.
bool SystolicConv1x1Nchw(char accelerator_mode, int64_t batch,
                        int64_t input_h, int64_t input_w, int64_t input_channels,
                        int64_t output_channels, const float* input,
                        const float* weights, const float* bias,
                        float* output, bool relu);

// Elementwise (in1 * in1_scale + in2 * in2_scale) / out_scale.
// Mode 2 uses the Gemmini accumulator and optional store-side ReLU; other
// modes use the CPU reference. Buffers contain dim contiguous FP32 elements.
// Exact in-place output is supported; partial buffer overlap is not.
void SystolicAdd
MLASCALL(char accelerator_mode, bool relu, const float* in1, float in1_scale,
         const float* in2, float in2_scale, float* out, float out_scale,
         size_t dim);

// NHWC/HWIO rectangular WS convolution; false means no instructions issued.
bool SystolicConvRect(char accelerator_mode, int64_t batch,
                     int64_t input_h, int64_t input_w, int64_t input_channels,
                     int64_t output_channels, int64_t output_h, int64_t output_w,
                     int64_t stride, int64_t padding, int64_t kernel,
                     const float* input, const float* weights, const float* bias,
                     float* output, bool relu, float output_scale);


void SystolicMultiply
MLASCALL(char accelerator_mode, bool relu, int dimI, int dimJ, int dimK, const float* in1, const float* in2,
         float* out, float real_multiplier, const float* bias = nullptr);

void SystolicMultiply
MLASCALL(char accelerator_mode, bool relu,
                            int dimI, int dimJ, int dimK,
                            const float* in1, int strideIn1,
                            const float* in2, int strideIn2,
                            float* out, int strideOut,
                            float real_multiplier,
                            const float* bias, int strideBias, bool repeating_bias);

// Matches Blass Gemm signature 
void SystolicGemm(
   char accelerator_mode, 
    bool TransA,
    bool TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C);

void SystolicGemm(
   char accelerator_mode, 
    bool TransA,
    bool TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float beta,
    float* C,
    int ldc);

void SystolicConv
MLASCALL(char accelerator_mode, int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int padding, int kernel_dim,
        const float* input,
        const float* weights,
        const float* bias,
        float* output,
        bool relu,
        float output_scale,
        int pool_size = 0, int pool_stride = 0, int pool_padding = 0);

void SystolicConvTranspose(char accelerator_mode, int batch_size, int in_dim, int in_channels,
                  int out_channels, int out_dim,
                  int stride, int padding, int kernel_dim,
                  const float* input,
                  const float* weights,
                  const float* bias,
                  float* output,
                  bool relu,
                  float output_scale);

void SystolicConvBackpropFilter(char accelerator_mode, int batch_size, int in_dim, int in_channels,
                  int out_channels, int out_dim,
                  int stride, int padding, int kernel_dim,
                  const float* input,
                  const float* weights,
                  const float* bias,
                  float* output,
                  bool relu,
                  float output_scale);
                            
#endif

#if defined(__riscv) && defined(SYSTOLIC_FP32) && !defined(SYSTOLIC_FP16)
// FP32 DIM16 correlation gather, implemented in systolic.cpp / MLAS.
// DPVO_CORR_GATHER gates the runner call site, not this shared MLAS API.
// These routines issue CGR1 instructions only when explicitly called. Probe is
// permitted only with a matching bitstream; an older bitstream may hang.
namespace systolic_corr {
// One 8x8 window in a contiguous FP32 [128,H,W] feature tensor.
// x/y are signed integer window origins, not floating-point sample centers.
struct Window {
  const float* data;
  uint32_t height, width;
  int32_t x, y;
};
struct GatherStatistics {
  uint64_t cycles, useful_bytes, strips, rows, read_stalls, write_stalls;
};

void Probe();
// Caller validates tensor/coordinate bounds and opts into matching CGR1 hardware.
// Load A[1,128], gather both windows into native B tiles, wait and check status,
// then return this job's counters. Throws on error; partial B must not be used.
// Gather/DotPreloaded require exclusive Gemmini use; do not interleave operations.
GatherStatistics Gather(const float* a, const Window& level0, const Window& level1);
// Consume the A/B prepared by Gather -> 128 FP32 dots; restore loop SPAD bounds.
void DotPreloaded(float* output);
}  // namespace systolic_corr
#endif
