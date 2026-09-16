// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

// FP32 NHWC/HWIO, WS, group=1, square kernel, unit dilation, no pooling.
// The encoder matches LoopConv.scala's rectangular CONFIG_1..6 ABI.
// Kept independent of RoCC so the exact command stream can be tested on host.
namespace systolic_rect {
struct Shape {
  int64_t batch, ih, iw, ci, co, oh, ow, kernel, stride, pad;
};
struct Tile { int rows, cols, ci, co; };

inline int64_t Blocks(int64_t n, int dim) { return (n + dim - 1) / dim; }

inline bool Fits(const Shape& s, const Tile& t, int dim,
                 int sp_rows, int acc_rows) {
  // Same conservative input footprint as LoopConvDerivedParams (includes the
  // unused trailing stride positions). Reserve half the memories for each loop.
  const int64_t ir = t.rows * s.stride + s.kernel - 1;
  const int64_t ic = t.cols * s.stride + s.kernel - 1;
  const int64_t a = Blocks(t.ci, dim) * ir * ic;
  const int64_t b = Blocks(t.co, dim) * s.kernel * s.kernel * t.ci;
  const int64_t c = Blocks(t.co, dim) * t.rows * t.cols;
  return a + b <= sp_rows / 2 && c <= acc_rows / 2;
}

inline bool Supported(const Shape& s) {
  for (int64_t v : {s.batch, s.ih, s.iw, s.ci, s.co, s.oh, s.ow})
    if (v <= 0 || v > 65535) return false;
  if (s.kernel <= 0 || s.kernel > 15 || s.stride <= 0 || s.stride > 255 ||
      s.pad < 0 || s.pad >= s.kernel) return false;
  if (s.ih + 2 * s.pad < s.kernel || s.iw + 2 * s.pad < s.kernel) return false;
  if (s.oh != (s.ih + 2 * s.pad - s.kernel) / s.stride + 1 ||
      s.ow != (s.iw + 2 * s.pad - s.kernel) / s.stride + 1) return false;
  const uint64_t max_elements = std::numeric_limits<size_t>::max() / sizeof(float);
  return uint64_t(s.batch) * s.ih * s.iw * s.ci <= max_elements &&
         uint64_t(s.batch) * s.oh * s.ow * s.co <= max_elements;
}

inline Tile FixedTile(const Shape& s, int dim, int sp_rows, int acc_rows) {
  Tile t{1, int(std::min<int64_t>(dim, s.ow)),
         int(std::min<int64_t>(dim, s.ci)), int(std::min<int64_t>(dim, s.co))};
  while (!Fits(s, t, dim, sp_rows, acc_rows)) {
    if (t.cols > 1) --t.cols;
    else if (t.ci > 1) --t.ci;
    else return {0, 0, 0, 0};
  }
  return t;
}

inline double LoopCount(const Shape& s, const Tile& t) {
  return double(s.batch) * Blocks(s.oh, t.rows) * Blocks(s.ow, t.cols) *
         Blocks(s.ci, t.ci) * Blocks(s.co, t.co);
}

inline Tile SelectTile(const Shape& s, int dim, int sp_rows, int acc_rows) {
  const Tile fixed = FixedTile(s, dim, sp_rows, acc_rows);
  if (!fixed.rows) return fixed;
  Tile best = fixed;
  double best_loops = LoopCount(s, best);
  double best_loads = std::numeric_limits<double>::infinity();
  auto next = [](int v, int64_t limit) { return int(v == limit ? limit + 1 : std::min<int64_t>(2 * v, limit)); };
  // Power-of-two growth plus exact dimension tails bounds the search. Retain
  // the stage-1 tile as a candidate; reserve half of each on-chip memory.
  for (int r = 1; r <= s.oh; r = next(r, s.oh))
    for (int c = fixed.cols; c <= s.ow; c = next(c, s.ow))
      for (int ci = fixed.ci; ci <= s.ci; ci = next(ci, s.ci))
        for (int co = fixed.co; co <= s.co; co = next(co, s.co)) {
          const Tile t{r, c, ci, co};
          if (!Fits(s, t, dim, sp_rows, acc_rows)) continue;
          const double loops = LoopCount(s, t);
          // Logical load estimate, not measured DMA traffic. Break equal loop
          // counts in favor of less repeated input/weight loading.
          const double loads = loops * (
              double(r * s.stride + s.kernel - 1) * (c * s.stride + s.kernel - 1) * ci +
              double(s.kernel) * s.kernel * ci * co);
          if (loops < best_loops || (loops == best_loops && loads < best_loads)) {
            best = t; best_loops = loops; best_loads = loads;
          }
        }
  return best;
}

// emit(funct, rs1, rs2): one custom instruction, all operands in element units
// except the four host addresses. No input/weight residency reuse is assumed.
template <class Emit>
inline void Run(const Shape& s, const Tile& t, const float* input,
                const float* weights, const float* bias, float* output,
                bool relu, Emit emit) {
  auto u = [](int64_t x) { return static_cast<uint64_t>(x); };
  auto address = [](const void* p) { return uint64_t(reinterpret_cast<uintptr_t>(p)); };
  for (int64_t n = 0; n < s.batch; ++n)
    for (int64_t y = 0; y < s.oh; y += t.rows)
      for (int64_t x = 0; x < s.ow; x += t.cols)
        for (int64_t oc = 0; oc < s.co; oc += t.co)
          for (int64_t ic = 0; ic < s.ci; ic += t.ci) {
            const int64_t rows = std::min<int64_t>(t.rows, s.oh - y);
            const int64_t cols = std::min<int64_t>(t.cols, s.ow - x);
            const int64_t och = std::min<int64_t>(t.co, s.co - oc);
            const int64_t ich = std::min<int64_t>(t.ci, s.ci - ic);
            const int64_t iy = y * s.stride - s.pad;
            const int64_t ix = x * s.stride - s.pad;
            const int64_t ir = rows * s.stride + s.kernel - 1;
            const int64_t iw = cols * s.stride + s.kernel - 1;
            const int64_t top = std::max<int64_t>(0, -iy);
            const int64_t left = std::max<int64_t>(0, -ix);
            const int64_t bottom = std::max<int64_t>(0, iy + ir - s.ih);
            const int64_t right = std::max<int64_t>(0, ix + iw - s.iw);
            const float* a = input + ((n * s.ih + iy + top) * s.iw + ix + left) * s.ci + ic;
            const float* b = weights + ic * s.co + oc;
            // Nonzero D initializes the accumulator. With no_bias=1 the RTL
            // synthesizes zeros rather than dereferencing the sentinel.
            const uint64_t d = ic ? 0 : (bias ? address(bias + oc) : 1);
            const uint64_t c = ic + ich == s.ci
                ? address(output + ((n * s.oh + y) * s.ow + x) * s.co + oc) : 0;
            emit(16, (u(s.co) << 48) | (u(s.ci) << 32) | (u(s.ih) << 16) | u(s.batch),
                 (u(s.pad) << 56) | (u(s.stride) << 48) | (u(s.ow) << 32) | (u(s.oh) << 16) | u(s.oh));
            emit(17, (u(s.kernel) << 48) | (u(s.ow) << 32) | (uint64_t(1) << 16) | (uint64_t(1) << 8),
                 (uint64_t(1) << 48) | (u(rows) << 32) | (u(cols) << 16) | u(och));
            emit(18, (u(s.kernel) << 48) | (u(s.kernel) << 32) | (u(ich) << 16) | u(left),
                 (u(right) << 48) | (u(top) << 32) | (u(bottom) << 24) | u(s.iw));
            emit(19, (u(rows) << 48) | 1,  // kernel_dilation=1, no pool padding
                 (u(s.ci) << 48) | (u(s.co) << 32) | (u(s.co) << 16) | u(cols));
            emit(20, address(b), c);
            emit(21, d, address(a));
            // max_pixels_per_row=1; no reuse, transpose, downsample or HW im2col.
            emit(15, (uint64_t(1) << 8) | (bias == nullptr), (uint64_t(relu) << 3) | 1);
          }
}
}  // namespace systolic_rect
