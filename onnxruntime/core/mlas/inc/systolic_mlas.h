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
