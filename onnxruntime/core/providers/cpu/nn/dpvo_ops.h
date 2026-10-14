// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <vector>
#include "core/framework/op_kernel.h"

namespace onnxruntime {
constexpr const char* kDpvoDomain = "dpvo";

// Shared validation and CPU references. These functions never submit Gemmini
// work. The Systolic kernels select hardware only after validation succeeds.
namespace dpvo {
struct SoftAggInputs {
  const float* x = nullptr;
  const int64_t* q = nullptr;
  std::array<const float*, 6> parameters{};
  float* output = nullptr;
  size_t batches = 0, edges = 0, channels = 0, groups = 0, max_group = 0;
  std::vector<size_t> offsets, edge_ids;
};
Status ReadSoftAggInputs(OpKernelContext* context, SoftAggInputs& inputs);
void SoftAggCpu(const SoftAggInputs& inputs, concurrency::ThreadPool* thread_pool,
                const std::array<bool, 3>& prepack_weights);

struct CorrelationInputs {
  const float* a = nullptr;
  const int64_t* windows = nullptr;
  float* dots = nullptr;
  uint64_t* counters = nullptr;
  size_t jobs = 0, channels = 0;
};
// windows[N,10] contains two {address,H,W,x,y} records per job. Addresses
// borrow live contiguous FP32 feature tensors for this synchronous Run only.
// This private ABI is not a serializable ONNX tensor representation.
Status ReadCorrelationInputs(OpKernelContext* context, CorrelationInputs& inputs);
void PackCorrelationWindow(const CorrelationInputs& inputs, size_t job, float* b);
void CorrelationCpu(const CorrelationInputs& inputs);
bool CorrelationGatherSupported(const CorrelationInputs& inputs);
}  // namespace dpvo
}  // namespace onnxruntime
