// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_SYSTOLIC
#include "core/providers/cpu/nn/dpvo_ops.h"
#include "core/common/replay_profile.h"
#include "core/mlas/inc/mlas.h"
#include <algorithm>
#include <limits>
#include <numeric>

namespace onnxruntime {
namespace dpvo {

Status ReadSoftAggInputs(OpKernelContext* context, SoftAggInputs& in) {
  const auto* x = context->Input<Tensor>(0);
  const auto* q = context->Input<Tensor>(1);
  const auto& shape = x->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 3 && shape[0] >= 1 &&
                        shape[1] >= 0 && shape[2] > 0,
                    "softagg_onchip expects X[B,E,C], B>=1, C>0");
  ORT_RETURN_IF_NOT(q->Shape().NumDimensions() == 1 && q->Shape()[0] == shape[1],
                    "softagg_onchip expects shared int64 inverse q[E]");
  in.batches = static_cast<size_t>(shape[0]);
  in.edges = static_cast<size_t>(shape[1]);
  in.channels = static_cast<size_t>(shape[2]);
  constexpr size_t limit = std::numeric_limits<size_t>::max() / sizeof(float);
  ORT_RETURN_IF_NOT(in.channels <= limit && in.edges <= limit / in.channels &&
                        (!in.edges || in.batches <= limit / (in.edges * in.channels)),
                    "softagg_onchip tensor byte count overflow");
  in.x = x->Data<float>();
  in.q = q->Data<int64_t>();
  for (size_t i = 0; i < in.parameters.size(); ++i) {
    const auto* parameter = context->Input<Tensor>(static_cast<int>(i + 2));
    const TensorShape expected = i % 2 == 0
        ? TensorShape({shape[2], shape[2]}) : TensorShape({shape[2]});
    ORT_RETURN_IF_NOT(parameter->Shape() == expected,
                      "softagg_onchip expects W[C,C] and bias[C]");
    in.parameters[i] = parameter->Data<float>();
  }
  for (size_t e = 0; e < in.edges; ++e) {
    ORT_RETURN_IF_NOT(in.q[e] >= 0 && static_cast<uint64_t>(in.q[e]) < in.edges,
                      "softagg_onchip q must be a Unique inverse in [0,E)");
    in.groups = std::max(in.groups, static_cast<size_t>(in.q[e]) + 1);
  }
  in.offsets.assign(in.groups + 1, 0);
  for (size_t e = 0; e < in.edges; ++e) ++in.offsets[static_cast<size_t>(in.q[e]) + 1];
  std::partial_sum(in.offsets.begin(), in.offsets.end(), in.offsets.begin());
  std::vector<size_t> cursors(in.offsets.begin(), in.offsets.end() - 1);
  in.edge_ids.resize(in.edges);
  for (size_t e = 0; e < in.edges; ++e)
    in.edge_ids[cursors[static_cast<size_t>(in.q[e])]++] = e;
  for (size_t g = 0; g < in.groups; ++g)
    in.max_group = std::max(in.max_group, in.offsets[g + 1] - in.offsets[g]);
  in.output = context->Output(0, shape)->MutableData<float>();
  return Status::OK();
}

void SoftAggCpu(const SoftAggInputs& in, concurrency::ThreadPool* thread_pool,
                const std::array<bool, 3>& prepack_weights) {
  if (!in.edges) return;
  ort_replay::Scope profile("kernel", "softagg.cpu_fallback", "SoftAggResidual", "CPU");
  const size_t c = in.channels, ec = in.edges * c, gc = in.groups * c;
  std::vector<float> scores(ec), values(ec), y(gc), h(gc);
  std::vector<float> exponentials(in.max_group * c), maximum(c), denominator(c);
  // CPU MatMul pre-packs constant weights. MLAS packed and unpacked GEMM use
  // different K blocking (256 versus 128 in this build), so the choice affects
  // FP32 rounding. Follow the same PrePack notification, including the session
  // option that disables prepacking. Pack only on CPU fallback; hardware keeps
  // the original weights and does not pay for duplicate packed allocations.
  std::array<std::vector<float>, 3> packed_storage;
  std::array<const float*, 3> packed{};
  const size_t packed_bytes = MlasGemmPackBSize(c, c);
  const size_t alignment = MlasGetPreferredBufferAlignment();
  for (size_t i = 0; i < packed.size(); ++i) {
    if (!prepack_weights[i] || !packed_bytes) continue;
    packed_storage[i].resize((packed_bytes + alignment - 1 + sizeof(float) - 1) / sizeof(float));
    const uintptr_t base = reinterpret_cast<uintptr_t>(packed_storage[i].data());
    float* aligned = reinterpret_cast<float*>((base + alignment - 1) / alignment * alignment);
    MlasGemmPackB(CblasNoTrans, c, c, in.parameters[2 * i], c, aligned);
    packed[i] = aligned;
  }
  const auto matmul = [&](size_t weight, size_t rows, const float* a, float* output) {
    MLAS_SGEMM_DATA_PARAMS parameters;
    parameters.A = a; parameters.lda = c;
    parameters.BIsPacked = packed[weight] != nullptr;
    parameters.B = parameters.BIsPacked ? packed[weight] : in.parameters[2 * weight];
    parameters.ldb = c;
    parameters.C = output; parameters.ldc = c;
    MlasGemm(CblasNoTrans, CblasNoTrans, rows, c, c, parameters, thread_pool);
  };
  for (size_t b = 0; b < in.batches; ++b) {
    const float* x = in.x + b * ec;
    matmul(0, in.edges, x, scores.data());
    matmul(1, in.edges, x, values.data());
    // Keep post-GEMM Add separate, as in the unfused model.
    for (size_t i = 0; i < ec; ++i) {
      scores[i] = in.parameters[1][i % c] + scores[i];
      values[i] = in.parameters[3][i % c] + values[i];
    }
    std::fill(y.begin(), y.end(), 0.0f);
    for (size_t g = 0; g < in.groups; ++g) {
      const size_t begin = in.offsets[g], end = in.offsets[g + 1];
      std::fill(maximum.begin(), maximum.end(), -std::numeric_limits<float>::infinity());
      std::fill(denominator.begin(), denominator.end(), 0.0f);
      for (size_t i = begin; i < end; ++i)
        for (size_t ch = 0; ch < c; ++ch) {
          const float s = scores[in.edge_ids[i] * c + ch];
          if (s > maximum[ch]) maximum[ch] = s;
        }
      for (size_t i = begin; i < end; ++i)
        for (size_t ch = 0; ch < c; ++ch)
          exponentials[(i - begin) * c + ch] = scores[in.edge_ids[i] * c + ch] - maximum[ch];
      MlasComputeExp(exponentials.data(), exponentials.data(), (end - begin) * c);
      for (size_t i = begin; i < end; ++i)
        for (size_t ch = 0; ch < c; ++ch)
          denominator[ch] += exponentials[(i - begin) * c + ch];
      for (size_t i = begin; i < end; ++i)
        for (size_t ch = 0; ch < c; ++ch) {
          const float weight = exponentials[(i - begin) * c + ch] / denominator[ch];
          const float product = values[in.edge_ids[i] * c + ch] * weight;
          y[g * c + ch] += product;
        }
    }
    matmul(2, in.groups, y.data(), h.data());
    for (size_t i = 0; i < gc; ++i) h[i] = in.parameters[5][i % c] + h[i];
    for (size_t e = 0; e < in.edges; ++e)
      for (size_t ch = 0; ch < c; ++ch)
        in.output[b * ec + e * c + ch] = x[e * c + ch] + h[static_cast<size_t>(in.q[e]) * c + ch];
  }
}

Status ReadCorrelationInputs(OpKernelContext* context, CorrelationInputs& in) {
  const auto* a = context->Input<Tensor>(0);
  const auto* windows = context->Input<Tensor>(1);
  const auto& shape = a->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 2 && shape[0] >= 0 && shape[1] > 0 &&
                        shape[1] <= std::numeric_limits<int>::max(),
                    "correlation_batch expects A[N,C], C in [1,INT_MAX]");
  ORT_RETURN_IF_NOT(windows->Shape() == TensorShape({shape[0], 10}),
                    "correlation_batch expects windows[N,10]");
  in.jobs = static_cast<size_t>(shape[0]);
  in.channels = static_cast<size_t>(shape[1]);
  in.a = a->Data<float>();
  in.windows = windows->Data<int64_t>();
  constexpr size_t limit = std::numeric_limits<size_t>::max() / sizeof(float);
  ORT_RETURN_IF_NOT(in.channels <= limit / 128 && in.jobs <= limit / 128 &&
                        in.jobs <= limit / in.channels,
                    "correlation_batch tensor byte count overflow");
  // Validate every borrowed feature view before the first job can execute.
  for (size_t j = 0; j < in.jobs; ++j)
    for (size_t level = 0; level < 2; ++level) {
      const int64_t* w = in.windows + j * 10 + level * 5;
      const uint64_t address = static_cast<uint64_t>(w[0]);
      ORT_RETURN_IF_NOT(w[1] >= 0 && w[2] >= 0,
                        "correlation_batch feature dimensions must be nonnegative");
      const uint64_t h = static_cast<uint64_t>(w[1]), width = static_cast<uint64_t>(w[2]);
      uint64_t bytes = 0;
      if (h && width) {
        ORT_RETURN_IF_NOT(width <= limit / in.channels && h <= limit / (width * in.channels),
                          "correlation_batch feature byte count overflow");
        bytes = h * width * in.channels * sizeof(float);
      }
      ORT_RETURN_IF_NOT((!bytes || (address != 0 && address % alignof(float) == 0)) &&
                            address <= std::numeric_limits<uintptr_t>::max(),
                        "correlation_batch requires aligned live nonempty feature addresses");
      ORT_RETURN_IF_NOT(bytes <= std::numeric_limits<uintptr_t>::max() - address,
                        "correlation_batch feature address overflow");
      ORT_RETURN_IF_NOT(w[3] <= std::numeric_limits<int64_t>::max() - 7 &&
                            w[4] <= std::numeric_limits<int64_t>::max() - 7,
                        "correlation_batch window origin overflow");
    }
  in.dots = context->Output(0, TensorShape({shape[0], 128}))->MutableData<float>();
  in.counters = context->Output(1, TensorShape({shape[0], 7}))->MutableData<uint64_t>();
  std::fill_n(in.counters, in.jobs * 7, UINT64_C(0));
  return Status::OK();
}

void PackCorrelationWindow(const CorrelationInputs& in, size_t job, float* b) {
  for (size_t channel = 0; channel < in.channels; ++channel)
    for (size_t level = 0; level < 2; ++level) {
      const int64_t* w = in.windows + job * 10 + level * 5;
      const float* features = reinterpret_cast<const float*>(static_cast<uintptr_t>(w[0]));
      for (int64_t gy = 0; gy < 8; ++gy)
        for (int64_t gx = 0; gx < 8; ++gx) {
          const int64_t x = w[3] + gx, y = w[4] + gy;
          b[channel * 128 + level * 64 + static_cast<size_t>(gy * 8 + gx)] =
              x >= 0 && y >= 0 && x < w[2] && y < w[1]
              ? features[(channel * static_cast<size_t>(w[1]) + static_cast<size_t>(y)) *
                         static_cast<size_t>(w[2]) + static_cast<size_t>(x)] : 0.0f;
        }
    }
}

void CorrelationCpu(const CorrelationInputs& in) {
  if (!in.jobs) return;
  std::vector<float> b(in.channels * 128);
  for (size_t j = 0; j < in.jobs; ++j) {
    PackCorrelationWindow(in, j, b.data());
    for (size_t column = 0; column < 128; ++column) {
      double sum = 0.0;
      for (size_t k = 0; k < in.channels; ++k)
        sum += static_cast<double>(in.a[j * in.channels + k]) * static_cast<double>(b[k * 128 + column]);
      in.dots[j * 128 + column] = static_cast<float>(sum);
    }
  }
}

bool CorrelationGatherSupported(const CorrelationInputs& in) {
  if (in.channels != 128 || !in.jobs) return false;
  constexpr uint64_t va_mask = (UINT64_C(1) << 40) - 1;
  for (size_t j = 0; j < in.jobs; ++j)
    for (size_t level = 0; level < 2; ++level) {
      const int64_t* w = in.windows + j * 10 + level * 5;
      if (w[1] <= 0 || w[2] <= 0 || w[1] > 65535 || w[2] > 65535 ||
          w[3] < std::numeric_limits<int32_t>::min() ||
          w[4] < std::numeric_limits<int32_t>::min() ||
          w[3] > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) - 7 ||
          w[4] > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) - 7) return false;
      const uint64_t bytes = UINT64_C(128) * static_cast<uint64_t>(w[1]) *
                             static_cast<uint64_t>(w[2]) * sizeof(float);
      if (bytes > va_mask - (static_cast<uint64_t>(w[0]) & va_mask)) return false;
    }
  return true;
}
}  // namespace dpvo

class SoftAggCpu final : public OpKernel {
 public:
  explicit SoftAggCpu(const OpKernelInfo& info) : OpKernel(info) {}
  Status PrePack(const Tensor& tensor, int input_index, bool& is_packed) override {
    // Retain raw inputs. Record the CPU MatMul packing choice and defer the
    // actual allocation to Compute, before any output writes.
    is_packed = false;
    if (input_index >= 2 && input_index <= 6 && input_index % 2 == 0 &&
        tensor.Shape().NumDimensions() == 2)
      prepack_weights_[static_cast<size_t>((input_index - 2) / 2)] = true;
    return Status::OK();
  }
  Status Compute(OpKernelContext* context) const override {
    dpvo::SoftAggInputs inputs;
    ORT_RETURN_IF_ERROR(dpvo::ReadSoftAggInputs(context, inputs));
    dpvo::SoftAggCpu(inputs, context->GetOperatorThreadPool(), prepack_weights_);
    return Status::OK();
  }
 private:
  std::array<bool, 3> prepack_weights_{};
};

class CorrelationCpu final : public OpKernel {
 public:
  explicit CorrelationCpu(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override {
    dpvo::CorrelationInputs inputs;
    ORT_RETURN_IF_ERROR(dpvo::ReadCorrelationInputs(context, inputs));
    dpvo::CorrelationCpu(inputs);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(softagg_onchip, kDpvoDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), SoftAggCpu);
ONNX_OPERATOR_KERNEL_EX(correlation_batch, kDpvoDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), CorrelationCpu);
}  // namespace onnxruntime
#endif  // USE_SYSTOLIC
