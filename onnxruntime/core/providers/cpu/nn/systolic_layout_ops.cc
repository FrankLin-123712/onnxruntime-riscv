// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CPU-only operators introduced by the Systolic graph transformer. These
// preserve NHWC buffers and combine adjacent elementwise work; they do not
// issue accelerator instructions.
#ifdef USE_SYSTOLIC
#include "core/providers/cpu/math/element_wise_ops.h"

#include <cmath>

namespace onnxruntime {

class AddRelu final : public OpKernel {
 public:
  explicit AddRelu(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    // Match Add's multidirectional broadcasting and Relu's Eigen maximum,
    // while writing each output element only once.
    ProcessBroadcastSpanFuncs funcs{
        [](BroadcastHelper& helper) {
          helper.OutputEigen<float>() =
              (helper.ScalarInput0<float>() + helper.EigenInput1<float>().array()).cwiseMax(0.0f);
        },
        [](BroadcastHelper& helper) {
          helper.OutputEigen<float>() =
              (helper.EigenInput0<float>().array() + helper.ScalarInput1<float>()).cwiseMax(0.0f);
        },
        [](BroadcastHelper& helper) {
          helper.OutputEigen<float>() =
              (helper.EigenInput0<float>() + helper.EigenInput1<float>()).cwiseMax(0.0f);
        }};
    UntypedBroadcastTwo(*context, funcs, 2.0);
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    AddRelu, 1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    AddRelu);

class InstanceNormNhwc final : public OpKernel {
 public:
  explicit InstanceNormNhwc(const OpKernelInfo& info)
      : OpKernel(info),
        epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-5f)),
        relu_(info.GetAttrOrDefault<int64_t>("relu", 0) != 0) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const auto* scale = context->Input<Tensor>(1);
    const auto* bias = context->Input<Tensor>(2);
    const auto& shape = input->Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() == 4,
                      "InstanceNormalization_nhwc requires rank-4 NHWC input");
    const int64_t channels = shape.GetDims()[3];
    ORT_RETURN_IF_NOT(scale->Shape().NumDimensions() == 1 &&
                          scale->Shape().Size() == channels,
                      "InstanceNormalization_nhwc scale must have shape [C]");
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1 &&
                          bias->Shape().Size() == channels,
                      "InstanceNormalization_nhwc bias must have shape [C]");
    auto* output = context->Output(0, shape);
    if (shape.Size() == 0) return Status::OK();

    const int64_t batches = shape.GetDims()[0];
    const int64_t spatial = shape.Slice(1, 3).Size();
    const int64_t sample_size = shape.SizeFromDimension(1);
    const float* scale_data = scale->Data<float>();
    const float* bias_data = bias->Data<float>();

    // Only O(C) temporary storage. Visit channels contiguously in NHWC for
    // each spatial position, retaining the spatial accumulation order of the
    // existing scalar NCHW kernel for each individual channel. In particular,
    // variance uses centered differences, not E[x*x] - E[x]*E[x].
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    Tensor workspace(DataTypeImpl::GetType<float>(), TensorShape({3, channels}), allocator);
    float* mean = workspace.MutableData<float>();
    float* variance_and_scale = mean + channels;
    float* shift = variance_and_scale + channels;
    for (int64_t n = 0; n < batches; ++n) {
      const float* x = input->Data<float>() + n * sample_size;
      float* y = output->MutableData<float>() + n * sample_size;
      for (int64_t c = 0; c < channels; ++c) mean[c] = x[c];
      for (int64_t s = 1; s < spatial; ++s) {
        const float* row = x + s * channels;
        for (int64_t c = 0; c < channels; ++c) mean[c] += row[c];
      }
      for (int64_t c = 0; c < channels; ++c) {
        mean[c] /= static_cast<float>(spatial);
        const float delta = x[c] - mean[c];
        variance_and_scale[c] = delta * delta;
      }
      for (int64_t s = 1; s < spatial; ++s) {
        const float* row = x + s * channels;
        for (int64_t c = 0; c < channels; ++c) {
          const float delta = row[c] - mean[c];
          variance_and_scale[c] += delta * delta;
        }
      }
      for (int64_t c = 0; c < channels; ++c) {
        const float inv_stdev = 1.0f / std::sqrt(
            variance_and_scale[c] / static_cast<float>(spatial) + epsilon_);
        variance_and_scale[c] = inv_stdev * scale_data[c];
        shift[c] = bias_data[c] - mean[c] * variance_and_scale[c];
      }
      const ConstEigenVectorArrayMap<float> channel_scale(variance_and_scale, channels);
      const ConstEigenVectorArrayMap<float> channel_shift(shift, channels);
      for (int64_t s = 0; s < spatial; ++s) {
        const ConstEigenVectorArrayMap<float> xi(x + s * channels, channels);
        EigenVectorArrayMap<float> yi(y + s * channels, channels);
        if (relu_) {
          yi = (xi * channel_scale + channel_shift).cwiseMax(0.0f);
        } else {
          yi = xi * channel_scale + channel_shift;
        }
      }
    }
    return Status::OK();
  }

 private:
  float epsilon_;
  bool relu_;
};

ONNX_CPU_OPERATOR_KERNEL(
    InstanceNormalization_nhwc, 1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    InstanceNormNhwc);

}  // namespace onnxruntime
#endif  // USE_SYSTOLIC
