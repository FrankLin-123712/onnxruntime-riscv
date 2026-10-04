// NHWC normalization ABI v1. Include after systolic_include.h or gemmini.h.
// Requires the opt-in FP32 DIM16 hardware / libgemmini_nhwc_norm.so.
#ifndef GEMMINI_NHWC_NORM_HELPER_H
#define GEMMINI_NHWC_NORM_HELPER_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__riscv) && (defined(GEMMINI_NHWC_NORM) || defined(HAS_NHWC_NORM))
#define GEMMINI_NHWC_NORM_HELPER_ENABLED 1

static inline void gemmini_nhwc_norm_barrier() {
  asm volatile("" ::: "memory");
}

static inline void gemmini_nhwc_norm_fence() {
  gemmini_fence();
  gemmini_nhwc_norm_barrier();
}

static inline void gemmini_nhwc_norm_context(size_t context, bool set_activation = false) {
  // CONFIG_NORM: act_msb=1, stats_id=context. No scalar-normalizer dependency.
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
      (uint64_t(context) << 8) | (uint64_t(1) << 16) |
      (uint64_t(!set_activation) << 17) | uint64_t(3), uint64_t(0), k_CONFIG);
}

static inline void gemmini_nhwc_norm_load_vector(const float* vector, size_t lanes,
                                                size_t context, unsigned command) {
  constexpr uint32_t parameter_row = uint32_t(ACC_ROWS - 1);
  constexpr uint32_t accumulator = uint32_t(1) << 31;
  gemmini_nhwc_norm_barrier();
  gemmini_nhwc_norm_context(context);
  gemmini_extended_mvin(vector, accumulator | parameter_row, lanes, 1);
  // Commands 5/6/7 import mean/alpha/shift; they do not write DRAM.
  gemmini_extended_mvout(const_cast<float*>(vector),
      accumulator | (uint32_t(command) << 26) | parameter_row, lanes, 1);
  gemmini_nhwc_norm_fence();
}
#endif

// Returns false before issuing instructions if the requested offload is
// unsupported. Caller owns the CPU fallback. Buffers are contiguous NHWC,
// gamma/beta contain C floats, and workspace is disjoint storage of >= 3*C
// floats. In-place input/output is supported. No tensor-sized temporary exists.
static inline bool gemmini_nhwc_instance_norm(
    const float* input, const float* gamma, const float* beta, float* output,
    size_t batches, size_t spatial, size_t channels, float epsilon, bool relu,
    float* workspace, size_t workspace_floats) {
#if !defined(GEMMINI_NHWC_NORM_HELPER_ENABLED)
  (void)input; (void)gamma; (void)beta; (void)output; (void)batches;
  (void)spatial; (void)channels; (void)epsilon; (void)relu;
  (void)workspace; (void)workspace_floats;
  return false;
#else
  static_assert(DIM == 16 && std::is_same<elem_t, float>::value &&
                std::is_same<acc_t, float>::value,
                "NHWC normalization ABI v1 requires FP32 DIM16");
  static_assert(ACC_ROWS > 4 * DIM, "NHWC normalization requires tile space");
  constexpr size_t contexts = 4;
  constexpr uint32_t accumulator = uint32_t(1) << 31;
  if (!batches || !spatial || !channels) return true;
  if (channels > contexts * DIM || !input || !gamma || !beta || !output || !workspace ||
      workspace_floats < 3 * channels || !std::isfinite(epsilon) || epsilon <= 0.0f ||
      spatial > std::numeric_limits<uint32_t>::max() ||
      spatial > std::numeric_limits<size_t>::max() / channels ||
      batches > std::numeric_limits<size_t>::max() / (spatial * channels) / sizeof(float))
    return false;

  const size_t groups = (channels + DIM - 1) / DIM;
  // Reserve one accumulator row for importing a CPU-finished parameter vector.
  // Both mvin and mvout command fields support at most DIM rows per command.
  size_t tile_rows = (ACC_ROWS - 1) / groups;
  if (tile_rows > 4095) tile_rows = 4095;
  tile_rows = (tile_rows / DIM) * DIM;
  float* const means = workspace;
  float* const alpha = workspace + channels;
  float* const shift = workspace + 2 * channels;
  const size_t batch_elements = spatial * channels;
  const float spatial_count = static_cast<float>(spatial);
  gemmini_nhwc_norm_barrier();
  gemmini_extended4_config_ld(channels * sizeof(float), MVIN_SCALE_IDENTITY, false, DIM, 0);
  gemmini_extended_config_st(channels * sizeof(float), relu ? 2 : 1, ACC_SCALE_IDENTITY);
  gemmini_nhwc_norm_context(0, true);  // activation 5 (affine) or 6 (affine + ReLU)

  for (size_t batch = 0; batch < batches; ++batch) {
    // Explicitly initialize every active stats context, even after other users.
    for (size_t c = 0; c < channels; ++c) means[c] = 0.0f;
    for (size_t group = 0; group < groups; ++group) {
      const size_t c = group * DIM;
      const size_t lanes = channels - c < DIM ? channels - c : DIM;
      gemmini_nhwc_norm_load_vector(means + c, lanes, group, 5);
    }

    for (unsigned pass = 0; pass < 3; ++pass) {
      for (size_t start = 0; start < spatial; start += tile_rows) {
        const size_t rows = spatial - start < tile_rows ? spatial - start : tile_rows;
        const bool last_tile = start + rows == spatial;
        for (size_t group = 0; group < groups; ++group) {
          const size_t c = group * DIM;
          const size_t lanes = channels - c < DIM ? channels - c : DIM;
          const uint32_t local = accumulator | uint32_t(group * tile_rows);
          const float* const source = input + batch * batch_elements + start * channels + c;
          for (size_t row = 0; row < rows; row += DIM) {
            const size_t load_rows = rows - row < DIM ? rows - row : DIM;
            gemmini_extended_mvin(source + row * channels, local + row, lanes, load_rows);
          }
          gemmini_nhwc_norm_context(group);
          if (pass == 2) {
            for (size_t row = 0; row < rows; row += DIM) {
              const size_t store_rows = rows - row < DIM ? rows - row : DIM;
              gemmini_extended_mvout(output + batch * batch_elements + (start + row) * channels + c,
                                     local + row, lanes, store_rows);
            }
          } else {
            const unsigned accumulate_command = pass == 0 ? 1 : 3;
            const size_t accumulate_rows = rows - size_t(last_tile);
            float* const stats = (pass == 0 ? means : alpha) + c;
            for (size_t row = 0; row < accumulate_rows; row += DIM) {
              const size_t store_rows = accumulate_rows - row < DIM ? accumulate_rows - row : DIM;
              gemmini_extended_mvout(stats, (local + row) | (accumulate_command << 26), lanes, store_rows);
            }
            if (last_tile)
              gemmini_extended_mvout(stats,
                  (local + rows - 1) | ((accumulate_command + 1) << 26), lanes, 1);
          }
        }
        // The next tile reuses these accumulator rows. Finish reads before mvin.
        gemmini_nhwc_norm_fence();
      }

      if (pass == 0) {
        for (size_t c = 0; c < channels; ++c) means[c] /= spatial_count;
        for (size_t group = 0; group < groups; ++group) {
          const size_t c = group * DIM;
          const size_t lanes = channels - c < DIM ? channels - c : DIM;
          gemmini_nhwc_norm_load_vector(means + c, lanes, group, 5);
        }
      } else if (pass == 1) {
        for (size_t c = 0; c < channels; ++c) {
          volatile float variance = alpha[c] / spatial_count;
          const float inverse_stddev = 1.0f / std::sqrt(variance + epsilon);
          alpha[c] = inverse_stddev * gamma[c];
          volatile float product = means[c] * alpha[c];
          shift[c] = beta[c] - product;
        }
        for (size_t group = 0; group < groups; ++group) {
          const size_t c = group * DIM;
          const size_t lanes = channels - c < DIM ? channels - c : DIM;
          gemmini_nhwc_norm_load_vector(alpha + c, lanes, group, 6);
          gemmini_nhwc_norm_load_vector(shift + c, lanes, group, 7);
        }
      }
    }
  }
  // Restore the ordinary store activation; no NHWC mode leaks into later ops.
  gemmini_extended_config_st(channels * sizeof(float), NO_ACTIVATION, ACC_SCALE_IDENTITY);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, uint64_t(3), uint64_t(0), k_CONFIG);
  gemmini_nhwc_norm_fence();
  return true;
#endif
}
#endif
