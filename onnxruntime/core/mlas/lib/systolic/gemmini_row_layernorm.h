// Row LayerNorm on the existing NHWC normalizer (HW-1 H0, no RTL change).
// Include after systolic_include.h. Requires the FP32 DIM16 NHWC
// normalization ABI v1 (C1 bitstream / libgemmini_softagg.so).
//
// y[r, :] = LayerNorm(x[r, :]) * gamma + beta [-> ReLU] for rows of C floats.
// The normalizer reduces along accumulator rows per lane, so each block of 16
// rows (edges) is transposed into the accumulator as C rows x 16 lanes with
// the WS array (A transposer, B = I). Per lane:
//   pass 0  SUM/MEAN       sum = x0 + x1 + ...        (CPU: mean = sum / C)
//   pass 1  VARIANCE       sumsq = sum (x - mean)^2   (CPU: sigma, alpha = 1/sigma)
//   pass 2  RESET          z = fl(fl(v * alpha) + shift) -> Z^T scratch
// and the WS array transposes Z^T back with B = diag(gamma chunk), so the
// product is fl(z * gamma); a stride-0 accumulating mvin adds beta and the
// store applies the optional ReLU.
//
// The statistics use the operations and order of the decomposed ONNX graph
// (ReduceMean, Sub, Pow, ReduceMean, Add, Sqrt). The normalizing step has no
// divider, so z is not bitwise equal to fl(d / sigma):
//   SUBTRACT: v = fl(x - mean) (stride-0 accumulating mvin of -mean), shift = -0,
//             z = fl(d * fl(1 / sigma))           (no cancellation)
//   AFFINE:   v = x, shift = -fl(mean * alpha)    (the plan's H0; cancels
//             when |mean| >> sigma)
// Signed zeros of x become +0 in the transpose. A block whose statistics are
// not finite (non-finite input or overflow) is reported for CPU fallback:
// in the transpose, an Inf/NaN of one row reaches every row of its block.
#ifndef GEMMINI_ROW_LAYERNORM_H
#define GEMMINI_ROW_LAYERNORM_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "gemmini_nhwc_norm.h"

enum GemminiRowLayerNormVariant : int {
  GEMMINI_ROW_LN_SUBTRACT = 0,
  GEMMINI_ROW_LN_AFFINE = 1,
};

struct GemminiRowLayerNormProfile {
  uint64_t blocks = 0, rounds = 0, fences = 0, commands = 0, fallback_blocks = 0;
  // CPU-side cycle split. Gemmini work overlaps issue and CPU math; fence
  // cycles are the CPU waiting for Gemmini.
  uint64_t setup_cycles = 0, issue_cycles = 0, cpu_cycles = 0, fence_cycles = 0,
           finish_cycles = 0;
};

// Floats of 64-byte aligned workspace for the given channel count.
static inline size_t gemmini_row_layernorm_workspace_floats(size_t channels) {
  // diag(gamma) C*16, Z^T 2*C*16, tail pad 16*C, identity 256, vectors 32*16.
  return channels * 16 * 4 + 256 + 32 * 16;
}

// Returns false before issuing instructions when unsupported. Otherwise every
// output row is written, except rows of blocks flagged in block_fallback
// (ceil(rows/16) bytes, 1 = caller must recompute rows [16b, 16b+16) on the
// CPU); their contents are unspecified. x and y must not overlap.
static inline bool gemmini_row_layernorm(
    const float* x, const float* gamma, const float* beta, float* y,
    size_t rows, size_t channels, float epsilon, bool relu, int variant,
    float* workspace, size_t workspace_floats, uint8_t* block_fallback,
    GemminiRowLayerNormProfile* profile) {
#if !defined(GEMMINI_NHWC_NORM_HELPER_ENABLED)
  (void)x; (void)gamma; (void)beta; (void)y; (void)rows; (void)channels;
  (void)epsilon; (void)relu; (void)variant; (void)workspace;
  (void)workspace_floats; (void)block_fallback; (void)profile;
  return false;
#else
  static_assert(DIM == 16 && std::is_same<elem_t, float>::value &&
                std::is_same<acc_t, float>::value,
                "Row LayerNorm requires FP32 DIM16");
  static_assert(BANK_NUM >= 3 && ACC_ROWS == 1024, "Row LayerNorm SP/ACC layout");
  constexpr uint32_t kAcc = uint32_t(1) << 31;
  constexpr uint32_t kAccumulate = uint32_t(1) << 30;
  constexpr uint32_t kParamRow = ACC_ROWS - 1;
  constexpr uint32_t kSpIdentity = 0;
  constexpr uint32_t kSpGamma = DIM;
  constexpr uint32_t kSpX = BANK_ROWS;      // bank 1: X chunks of two blocks
  constexpr uint32_t kSpZ = 2 * BANK_ROWS;  // bank 2: Z^T chunks of two blocks
  constexpr uint64_t kSum = 1, kMean = 2, kVariance = 3, kInvStd = 4, kLoadMean = 5,
                     kLoadAlpha = 6, kLoadShift = 7;

  if (rows == 0) return true;
  if (!x || !gamma || !beta || !y || !workspace || !block_fallback) return false;
  if (channels < DIM || channels % DIM != 0 || 2 * channels + DIM + 1 > ACC_ROWS ||
      2 * channels > BANK_ROWS || DIM + channels > BANK_ROWS)
    return false;
  if (!std::isfinite(epsilon) || epsilon <= 0.0f) return false;
  if (variant != GEMMINI_ROW_LN_SUBTRACT && variant != GEMMINI_ROW_LN_AFFINE) return false;
  if (workspace_floats < gemmini_row_layernorm_workspace_floats(channels) ||
      reinterpret_cast<uintptr_t>(workspace) % 64 != 0)
    return false;
  if (channels > UINT32_MAX / sizeof(float)) return false;

  auto now = []() -> uint64_t {
    uint64_t value;
    asm volatile("rdcycle %0" : "=r"(value));
    return value;
  };
  uint64_t mark = profile ? now() : 0;
  auto lap = [&](uint64_t& bucket) {
    if (!profile) return;
    const uint64_t t = now();
    bucket += t - mark;
    mark = t;
  };
  auto fence = [&]() {
    if (profile) {
      lap(profile->issue_cycles);
      ++profile->fences;
    }
    gemmini_nhwc_norm_fence();
    asm volatile("fence" ::: "memory");
    if (profile) lap(profile->fence_cycles);
  };
  uint64_t commands = 0;

  const size_t C = channels;
  const uint32_t chunks = static_cast<uint32_t>(C / DIM);
  const uint32_t region_rows = static_cast<uint32_t>(C);
  const uint32_t out_base = 2 * region_rows;
  const uint32_t out_tiles = (kParamRow - out_base) / DIM;
  const float divisor = static_cast<float>(C);  // ReduceMean divides by float(C)
  const size_t blocks = (rows + DIM - 1) / DIM;

  float* const diag = workspace;                 // [C][16]: diag(gamma) chunks
  float* const zt = diag + C * DIM;              // [2][C][16]: Z^T per region
  float* const pad = zt + 2 * C * DIM;           // [16][C]: zero-padded tail block
  float* const identity = pad + DIM * C;         // [16][16]
  float* const vectors = identity + DIM * DIM;   // 32 vectors of 16 floats
  auto vec = [&](unsigned index, unsigned region) { return vectors + (index * 2 + region) * DIM; };
  float* const zeros = vec(0, 0);
  float* const negative_zeros = vec(0, 1);
  // sums 1, sumsq 2, mean 3, -mean 4, alpha 5, shift 6 (x2 regions)

  for (size_t i = 0; i < DIM * DIM; ++i) identity[i] = (i / DIM == i % DIM) ? 1.0f : 0.0f;
  for (size_t c = 0; c < C; ++c)
    for (size_t j = 0; j < DIM; ++j) diag[c * DIM + j] = (c % DIM == j) ? gamma[c] : 0.0f;
  for (size_t i = 0; i < DIM; ++i) {
    zeros[i] = 0.0f;
    negative_zeros[i] = -0.0f;
  }
  const size_t tail = rows % DIM;
  if (tail) {
    std::memcpy(pad, x + (blocks - 1) * DIM * C, tail * C * sizeof(float));
    std::memset(pad + tail * C, 0, (DIM - tail) * C * sizeof(float));
  }
  for (size_t b = 0; b < blocks; ++b) block_fallback[b] = 0;
  lap(profile ? profile->setup_cycles : mark);

  asm volatile("fence" ::: "memory");
  // WS with the A transposer for both transposes. Loads: id0 X rows, id1
  // stride-0 broadcast rows (beta, -mean), id2 dense 16-float rows.
  gemmini_extended_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0, 0, 1, 1, 0);
  gemmini_extended4_config_ld(C * sizeof(float), MVIN_SCALE_IDENTITY, false, DIM, 0);
  gemmini_extended4_config_ld(0, MVIN_SCALE_IDENTITY, false, DIM, 1);
  gemmini_extended4_config_ld(DIM * sizeof(float), MVIN_SCALE_IDENTITY, false, DIM, 2);
  commands += 2 + 3;
  gemmini_extended_mvin3(identity, kSpIdentity, DIM, DIM);
  for (uint32_t j = 0; j < chunks; ++j)
    gemmini_extended_mvin3(diag + j * DIM * DIM, kSpGamma + j * DIM, DIM, DIM);
  commands += 1 + chunks;

  const auto norm_mode = [&]() {
    gemmini_extended_config_st(DIM * sizeof(float), 1, ACC_SCALE_IDENTITY);
    gemmini_nhwc_norm_context(0, true);  // activation 5: NHWC normalization
    commands += 2;
  };
  const auto store_mode = [&]() {
    gemmini_extended_config_st(C * sizeof(float), relu ? RELU : NO_ACTIVATION,
                               ACC_SCALE_IDENTITY);
    commands += 1;
  };
  const auto load_vector = [&](const float* vector, unsigned region, uint64_t command) {
    gemmini_nhwc_norm_context(region);
    gemmini_extended_mvin(vector, kAcc | kParamRow, DIM, 1);
    gemmini_extended_mvout(const_cast<float*>(vector), kAcc | (command << 26) | kParamRow, DIM, 1);
    commands += 3;
  };
  // Reduction pass over one region: rows 0..C-2 with `accumulate`, the last
  // row with `finish`, which writes the 16 lane results to `out` and clears
  // the sums.
  const auto reduce = [&](unsigned region, uint64_t accumulate, uint64_t finish, float* out) {
    const uint32_t base = region * region_rows;
    gemmini_nhwc_norm_context(region);
    for (uint32_t t = 0; t < chunks; ++t) {
      const uint32_t r = t + 1 == chunks ? DIM - 1 : DIM;
      gemmini_extended_mvout(out, kAcc | (accumulate << 26) | (base + t * DIM), DIM, r);
    }
    gemmini_extended_mvout(out, kAcc | (finish << 26) | (base + region_rows - 1), DIM, 1);
    commands += 2 + chunks;
  };

  norm_mode();
  for (unsigned region = 0; region < 2; ++region) {
    load_vector(zeros, region, kLoadMean);  // also clears the lane sums
    if (variant == GEMMINI_ROW_LN_SUBTRACT) load_vector(negative_zeros, region, kLoadShift);
  }

  // Blocks are processed two per round (regions 0/1, contexts 0/1). The exit
  // transpose of a round is issued after the next round's first fence, so it
  // overlaps that round's CPU work; its Z^T rows are complete by then.
  size_t previous_first = 0, previous_count = 0;
  uint32_t out_slot = 0;
  const auto exit_round = [&](size_t first, size_t count) {
    store_mode();
    for (size_t k = 0; k < count; ++k) {
      const size_t block = first + k;
      const size_t valid = block + 1 == blocks && tail ? tail : DIM;
      const uint32_t sp = kSpZ + static_cast<uint32_t>(k) * region_rows;
      const float* z = zt + k * C * DIM;
      for (uint32_t j = 0; j < chunks; ++j)
        gemmini_extended_mvin3(z + j * DIM * DIM, sp + j * DIM, DIM, DIM);
      for (uint32_t j = 0; j < chunks; ++j) {
        const uint32_t tile = out_base + (out_slot++ % out_tiles) * DIM;
        gemmini_extended_preload(kSpGamma + j * DIM, kAcc | tile, DIM, DIM, DIM, DIM);
        gemmini_extended_compute_preloaded(sp + j * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
        gemmini_extended_mvin2(beta + j * DIM, kAcc | kAccumulate | tile, DIM, DIM);
        gemmini_extended_mvout(y + block * DIM * C + j * DIM, kAcc | tile, DIM, valid);
      }
      commands += 5 * chunks;
    }
  };

  for (size_t first = 0; first < blocks; first += 2) {
    const size_t count = blocks - first < 2 ? blocks - first : 2;  // blocks this round
    if (profile) ++profile->rounds;
    // Entry transpose: ACC[region][c][lane e] = x[e][c]; then pass 0.
    for (size_t k = 0; k < count; ++k) {
      const size_t block = first + k;
      const float* source = block + 1 == blocks && tail ? pad : x + block * DIM * C;
      const uint32_t sp = kSpX + static_cast<uint32_t>(k) * region_rows;
      const uint32_t acc = static_cast<uint32_t>(k) * region_rows;
      for (uint32_t j = 0; j < chunks; ++j) {
        gemmini_extended_mvin(source + j * DIM, sp + j * DIM, DIM, DIM);
        if (j == 0) {
          gemmini_extended_preload(kSpIdentity, kAcc | acc, DIM, DIM, DIM, DIM);
          gemmini_extended_compute_preloaded(sp, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
        } else {
          gemmini_extended_preload(GARBAGE_ADDR, kAcc | (acc + j * DIM), DIM, DIM, DIM, DIM);
          gemmini_extended_compute_accumulated(sp + j * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
        }
      }
      commands += 3 * chunks;
      reduce(static_cast<unsigned>(k), kSum, kMean, vec(1, static_cast<unsigned>(k)));
    }
    fence();  // sums ready; previous round's Z^T written

    if (previous_count) {
      exit_round(previous_first, previous_count);
      lap(profile ? profile->issue_cycles : mark);
    }
    for (size_t k = 0; k < count; ++k) {
      const unsigned region = static_cast<unsigned>(k);
      const size_t valid = first + k + 1 == blocks && tail ? tail : DIM;
      const float* sums = vec(1, region);
      float* mean = vec(3, region);
      float* negative_mean = vec(4, region);
      for (size_t lane = 0; lane < DIM; ++lane) {
        const float sum = sums[lane];
        if (lane < valid && !std::isfinite(sum)) block_fallback[first + k] = 1;
        mean[lane] = sum / divisor;
        negative_mean[lane] = -mean[lane];
      }
    }
    lap(profile ? profile->cpu_cycles : mark);
    norm_mode();
    for (size_t k = 0; k < count; ++k) {
      const unsigned region = static_cast<unsigned>(k);
      load_vector(vec(3, region), region, kLoadMean);
      reduce(region, kVariance, kInvStd, vec(2, region));
    }
    fence();  // squared sums ready

    for (size_t k = 0; k < count; ++k) {
      const unsigned region = static_cast<unsigned>(k);
      const size_t valid = first + k + 1 == blocks && tail ? tail : DIM;
      const float* squares = vec(2, region);
      const float* mean = vec(3, region);
      float* alpha = vec(5, region);
      float* shift = vec(6, region);
      for (size_t lane = 0; lane < DIM; ++lane) {
        const float sumsq = squares[lane];
        if (lane < valid && !std::isfinite(sumsq)) block_fallback[first + k] = 1;
        const float variance = sumsq / divisor;
        const float sigma = std::sqrt(variance + epsilon);  // Add(eps), Sqrt
        alpha[lane] = 1.0f / sigma;
        volatile float centered = mean[lane] * alpha[lane];
        shift[lane] = -centered;
      }
    }
    lap(profile ? profile->cpu_cycles : mark);
    for (size_t k = 0; k < count; ++k) {
      const unsigned region = static_cast<unsigned>(k);
      const uint32_t base = region * region_rows;
      if (variant == GEMMINI_ROW_LN_SUBTRACT) {
        for (uint32_t t = 0; t < chunks; ++t)
          gemmini_extended_mvin2(vec(4, region), kAcc | kAccumulate | (base + t * DIM), DIM, DIM);
        commands += chunks;
      } else {
        load_vector(vec(6, region), region, kLoadShift);
      }
      load_vector(vec(5, region), region, kLoadAlpha);
      gemmini_nhwc_norm_context(region);
      for (uint32_t t = 0; t < chunks; ++t)
        gemmini_extended_mvout(zt + region * C * DIM + t * DIM * DIM,
                               kAcc | (base + t * DIM), DIM, DIM);
      commands += 1 + chunks;
    }
    previous_first = first;
    previous_count = count;
    if (profile) profile->blocks += count;
  }
  fence();
  exit_round(previous_first, previous_count);
  // Restore ordinary store/execute state; no NHWC mode or transpose leaks.
  gemmini_extended_config_st(DIM * sizeof(float), NO_ACTIVATION, ACC_SCALE_IDENTITY);
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, uint64_t(3), uint64_t(0), k_CONFIG);
  gemmini_extended_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0, 0, 1, 0, 0);
  commands += 4;
  fence();
  if (profile) {
    profile->commands += commands;
    for (size_t b = 0; b < blocks; ++b) profile->fallback_blocks += block_fallback[b];
    lap(profile->finish_cycles);
  }
  return true;
#endif
}

#endif  // GEMMINI_ROW_LAYERNORM_H
