// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/util/math_cpuonly.h"

namespace onnxruntime {

// Shared CPU reference/fallback. Workspace contains three channel vectors.
// Retain spatial accumulation order and centered variance for near-constant X.
inline void InstanceNormNhwcCpu(const float* input, const float* scale,
                                const float* bias, float* output,
                                int64_t batches, int64_t spatial, int64_t channels,
                                float epsilon, bool relu, float* workspace) {
  float* mean = workspace;
  float* variance_and_scale = mean + channels;
  float* shift = variance_and_scale + channels;
  const int64_t sample_size = spatial * channels;
  for (int64_t n = 0; n < batches; ++n) {
    const float* x = input + n * sample_size;
    float* y = output + n * sample_size;
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
          variance_and_scale[c] / static_cast<float>(spatial) + epsilon);
      variance_and_scale[c] = inv_stdev * scale[c];
      shift[c] = bias[c] - mean[c] * variance_and_scale[c];
    }
    const ConstEigenVectorArrayMap<float> channel_scale(variance_and_scale, channels);
    const ConstEigenVectorArrayMap<float> channel_shift(shift, channels);
    for (int64_t s = 0; s < spatial; ++s) {
      const ConstEigenVectorArrayMap<float> xi(x + s * channels, channels);
      EigenVectorArrayMap<float> yi(y + s * channels, channels);
      if (relu) yi = (xi * channel_scale + channel_shift).cwiseMax(0.0f);
      else yi = xi * channel_scale + channel_shift;
    }
  }
}
}  // namespace onnxruntime
