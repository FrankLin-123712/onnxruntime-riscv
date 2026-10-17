// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/mlas/inc/mlas.h"
#include "core/providers/systolic/gemmini_executor.h"
#include "core/providers/cpu/nn/dpvo_ops.h"
#include "core/common/replay_profile.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>

namespace onnxruntime {
namespace systolic {
#if defined(__riscv) && defined(SYSTOLIC_FP32) && !defined(SYSTOLIC_FP16)
namespace {
constexpr uint32_t kChannels = 384, kAcc = UINT32_C(0x80000000);
LocalBuffer Tile(uint32_t base, uint32_t height, uint32_t columns) {
  return {LocalMemory::Sp, LocalLayout::TileRows, base, height * ((columns + 15) / 16),
          height, height, columns};
}
LocalBuffer Acc(uint32_t base, uint32_t height, uint32_t columns) {
  return {LocalMemory::Acc, LocalLayout::ChannelBlocks, base, height * ((columns + 15) / 16),
          height, height, columns};
}
LocalBuffer Rows(uint32_t base, uint32_t extent) {
  return {LocalMemory::Sp, LocalLayout::Rows, base, extent, 1, extent, 16};
}
}  // namespace

GemminiExecutor::GemminiExecutor() { systolic_local::Fence(); }
GemminiExecutor::~GemminiExecutor() {
  systolic_local::Fence();
  if (allocator_.Pending()) allocator_.Complete(false);
}
void GemminiExecutor::WaitLocal() {
  systolic_local::Fence();
  ++waits_;
  if (allocator_.Pending()) allocator_.Complete(true);
}

void GemminiExecutor::LoadWeights(const float* weights, uint32_t column, LocalHandle dst) {
  LocalAllocator::Check(c384_initialized_, "Initialize C384 local configuration before this operation");
  const auto& d = allocator_.Describe(dst);
  LocalAllocator::Check(d.layout == LocalLayout::TileRows && d.rows == kChannels,
                        "weights require native K384 tile rows");
  Submit({}, {dst}, [&] { systolic_local::LoadWeights(weights, d.base, column, d.columns, traffic_); });
  WaitLocal();
}
void GemminiExecutor::LoadBias(const float* bias, uint32_t column, uint32_t panel, LocalHandle dst) {
  LocalAllocator::Check(c384_initialized_, "Initialize C384 local configuration before this operation");
  const auto& d = allocator_.Describe(dst);
  LocalAllocator::Check(d.memory == LocalMemory::Sp && d.layout == LocalLayout::Rows &&
                        panel && panel / 16 <= d.extent && panel % 16 == 0 &&
                        column <= kChannels && panel <= kChannels - column,
                        "bias panel exceeds local allocation");
  allocator_.Reshape(dst, panel / 16, 16);
  Submit({}, {dst}, [&] { systolic_local::LoadBias(bias, d.base, column, panel, traffic_); });
  WaitLocal();
}
void GemminiExecutor::LoadIndexed(const float* x, const std::vector<size_t>& ids,
                                 size_t begin, size_t end, LocalHandle dst) {
  LocalAllocator::Check(c384_initialized_, "Initialize C384 local configuration before this operation");
  LocalAllocator::Check(end >= begin && end - begin <= UINT32_MAX, "invalid indexed row count");
  LocalAllocator::Check(allocator_.Describe(dst).layout == LocalLayout::TileRows,
                        "indexed rows require native SP tiles");
  allocator_.Reshape(dst, static_cast<uint32_t>(end - begin), kChannels);
  Submit({}, {dst}, [&] {
    systolic_local::LoadIndexedRows(x, ids, begin, end, allocator_.Describe(dst).base, traffic_);
  });
  WaitLocal();
}
void GemminiExecutor::LoadGroups(const float* y, size_t begin, uint32_t count, LocalHandle dst) {
  LocalAllocator::Check(c384_initialized_, "Initialize C384 local configuration before this operation");
  LocalAllocator::Check(allocator_.Describe(dst).layout == LocalLayout::TileRows,
                        "group rows require native SP tiles");
  allocator_.Reshape(dst, count, kChannels);
  Submit({}, {dst}, [&] {
    systolic_local::LoadGroupRows(y, begin, count, allocator_.Describe(dst).base, traffic_);
  });
  WaitLocal();
}
void GemminiExecutor::LoadResidual(const float* input, LocalHandle dst) {
  LocalAllocator::Check(c384_initialized_, "Initialize C384 local configuration before this operation");
  const auto& d = allocator_.Describe(dst);
  LocalAllocator::Check(d.memory == LocalMemory::Sp && d.layout == LocalLayout::Rows &&
                        d.rows == 1 && d.columns == 16, "residual staging requires one SP row");
  Submit({}, {dst}, [&] { systolic_local::LoadRow(input, allocator_.Address(dst, 0), traffic_); });
  traffic_.x += 64;
  WaitLocal();
}

void GemminiExecutor::SubmitLocalGemm(LocalHandle a, LocalHandle weights, LocalHandle output) {
  LocalAllocator::Check(c384_initialized_, "Initialize C384 local configuration before this operation");
  const auto& x = allocator_.Describe(a);
  const auto& w = allocator_.Describe(weights);
  const auto& c = allocator_.Describe(output);
  LocalAllocator::Check(x.layout == LocalLayout::TileRows && w.layout == LocalLayout::TileRows &&
      c.layout == LocalLayout::ChannelBlocks && x.columns == w.rows && w.columns == c.columns &&
      x.rows <= c.pitch && x.columns % 16 == 0 && w.columns % 16 == 0,
      "incompatible explicit local GEMM layouts/shapes");
  allocator_.Reshape(output, x.rows, w.columns);
  Submit({a, weights}, {output}, [&] {
    systolic_local::MatmulLocal({x.base, w.base, c.base, x.rows, w.columns, x.columns,
        x.columns, w.columns, c.pitch}, traffic_);
  });
  ++local_gemms_;
  local_gemm_bytes_ += uint64_t(x.rows) * w.columns * sizeof(float);
}

void GemminiExecutor::Reduce(LocalHandle s, LocalHandle v, LocalHandle bias_s, LocalHandle bias_v,
    LocalHandle output, uint32_t row, uint32_t block, uint32_t count, bool residual) {
  const auto& sd = allocator_.Describe(s);
  const auto& vd = allocator_.Describe(v);
  const auto& bs_desc = allocator_.Describe(bias_s);
  const auto& bv_desc = allocator_.Describe(bias_v);
  const auto& od = allocator_.Describe(output);
  LocalAllocator::Check(bs_desc.memory == LocalMemory::Sp && bs_desc.layout == LocalLayout::Rows &&
      bv_desc.memory == LocalMemory::Sp && bv_desc.layout == LocalLayout::Rows &&
      od.memory == LocalMemory::Sp && od.layout == LocalLayout::Rows && od.rows == 1 && od.columns == 16 &&
      (!residual || (vd.memory == LocalMemory::Sp && vd.rows == 1 && vd.columns == 16)),
      "SoftAgg bias/staging requires SP vector rows");
  LocalAllocator::Check(sd.layout == LocalLayout::ChannelBlocks && count && row < sd.rows &&
      count <= sd.rows - row && (residual ? vd.layout == LocalLayout::Rows :
      vd.layout == LocalLayout::ChannelBlocks && row < vd.rows && count <= vd.rows - row),
      "SoftAgg group exceeds valid local rows");
  const uint32_t sa = kAcc | allocator_.Address(s, row, block);
  const uint32_t va = residual ? allocator_.Address(v, 0) : kAcc | allocator_.Address(v, row, block);
  const uint32_t bs = allocator_.Address(bias_s, block);
  const uint32_t bv = allocator_.Address(bias_v, residual ? 0 : block);
  const uint32_t dst = allocator_.Address(output, 0);
  // SAG1 mutates S for its exp/sum passes. Residual mode only reads H and X;
  // bias_v is prefetched even in residual mode, although its value is ignored.
  if (residual)
    Submit({s, v, bias_s, bias_v}, {output}, [&] {
      systolic_local::RunGroup(sa, va, bs, bv, count, true, dst, traffic_);
    });
  else
    Submit({s, v, bias_s, bias_v}, {s, output}, [&] {
      systolic_local::RunGroup(sa, va, bs, bv, count, false, dst, traffic_);
    }, true);
  // RunGroup polls SAG1 until the local write pipeline has drained.
  allocator_.Complete(true);
}
void GemminiExecutor::Store(LocalHandle source, float* output) {
  LocalAllocator::Check(c384_initialized_, "Initialize C384 local configuration before this operation");
  const auto& d = allocator_.Describe(source);
  LocalAllocator::Check(d.memory == LocalMemory::Sp && d.layout == LocalLayout::Rows &&
                        d.rows == 1 && d.columns == 16, "Store requires one SP vector row");
  Submit({source}, {}, [&] { systolic_local::StoreRow(output, allocator_.Address(source, 0), traffic_); });
  WaitExternal();
}

// The fused C1 plan leases one complete tile transaction. Intermediate staging
// rows never escape it. RunGroup's status and the dispatcher's normal drain
// order every previous mvout before the next staging overwrite, exactly as in
// C1. A final fence commits external writes before any lease can be released.
void GemminiExecutor::SpillGroups(LocalHandle s, LocalHandle v, LocalHandle bias_s, LocalHandle bias_v,
    LocalHandle output, const std::vector<size_t>& offsets, size_t first, size_t last,
    uint32_t column, float* y) {
  const auto& sd = allocator_.Describe(s);
  const auto& vd = allocator_.Describe(v);
  const auto& bs = allocator_.Describe(bias_s);
  const auto& bv = allocator_.Describe(bias_v);
  const auto& out = allocator_.Describe(output);
  LocalAllocator::Check(sd.layout == LocalLayout::ChannelBlocks && vd.layout == sd.layout &&
      sd.rows == vd.rows && sd.columns == vd.columns && sd.columns % 16 == 0 &&
      bs.layout == LocalLayout::Rows && bv.layout == bs.layout &&
      bs.rows >= sd.columns / 16 && bv.rows >= sd.columns / 16 &&
      out.memory == LocalMemory::Sp && out.layout == LocalLayout::Rows && out.rows == 1 &&
      first < last && last < offsets.size() && offsets[last] - offsets[first] == sd.rows &&
      column <= kChannels && sd.columns <= kChannels - column, "invalid SoftAgg spill tile");
  for (size_t g = first; g < last; ++g)
    LocalAllocator::Check(offsets[g] <= offsets[g + 1] && offsets[g + 1] - offsets[g] <= 128,
                          "invalid complete SoftAgg group");
  Submit({s, v, bias_s, bias_v}, {s, output}, [&] {
    for (size_t g = first; g < last; ++g) {
      const uint32_t n = static_cast<uint32_t>(offsets[g + 1] - offsets[g]);
      if (!n) continue;
      const uint32_t row = static_cast<uint32_t>(offsets[g] - offsets[first]);
      for (uint32_t j = 0; j < sd.columns / 16; ++j) {
        systolic_local::RunGroup(kAcc | (sd.base + j * sd.pitch + row),
            kAcc | (vd.base + j * vd.pitch + row), bs.base + j, bv.base + j,
            n, false, out.base, traffic_);
        systolic_local::StoreRow(y + g * kChannels + column + j * 16, out.base, traffic_);
        traffic_.y_write += 64;
      }
    }
  }, true);
  WaitExternal();
}
void GemminiExecutor::StoreResidualRows(LocalHandle h, LocalHandle residual,
    LocalHandle bias_s, LocalHandle bias_v, LocalHandle output, const std::vector<size_t>& offsets,
    const std::vector<size_t>& edge_ids, size_t first, uint32_t count, uint32_t column,
    const float* input, float* result) {
  const auto& hd = allocator_.Describe(h);
  const auto& xd = allocator_.Describe(residual);
  const auto& bs = allocator_.Describe(bias_s);
  const auto& bv = allocator_.Describe(bias_v);
  const auto& out = allocator_.Describe(output);
  LocalAllocator::Check(hd.layout == LocalLayout::ChannelBlocks && count == hd.rows &&
      hd.columns % 16 == 0 && bs.layout == LocalLayout::Rows && bv.layout == bs.layout &&
      bs.rows >= hd.columns / 16 && bv.rows && xd.memory == LocalMemory::Sp &&
      xd.layout == LocalLayout::Rows && xd.rows == 1 && out.memory == LocalMemory::Sp &&
      out.layout == LocalLayout::Rows && out.rows == 1 && first < offsets.size() &&
      count < offsets.size() - first && offsets[first + count] <= edge_ids.size() &&
      column <= kChannels && hd.columns <= kChannels - column, "invalid SoftAgg residual tile");
  Submit({h, bias_s, bias_v}, {residual, output}, [&] {
    for (uint32_t g = 0; g < count; ++g)
      for (size_t member = offsets[first + g]; member < offsets[first + g + 1]; ++member) {
        const size_t e = edge_ids[member];
        for (uint32_t j = 0; j < hd.columns / 16; ++j) {
          systolic_local::LoadRow(input + e * kChannels + column + j * 16, xd.base, traffic_);
          traffic_.x += 64;
          systolic_local::RunGroup(kAcc | (hd.base + j * hd.pitch + g), xd.base,
              bs.base + j, bv.base, 1, true, out.base, traffic_);
          systolic_local::StoreRow(result + e * kChannels + column + j * 16, out.base, traffic_);
          traffic_.output += 64;
        }
      }
  });
  WaitExternal();
}

systolic_corr::GatherStatistics GemminiExecutor::Gather(const float* a,
    const systolic_corr::Window& level0, const systolic_corr::Window& level1,
    LocalHandle local_a, LocalHandle local_b) {
  const auto& ad = allocator_.Describe(local_a);
  const auto& bd = allocator_.Describe(local_b);
  LocalAllocator::Check(ad.layout == LocalLayout::TileRows && ad.rows == 1 && ad.columns == 128 &&
      ad.extent >= 128 && bd.layout == LocalLayout::TileRows && bd.rows == 128 && bd.columns == 128 &&
      bd.extent >= 1024 && bd.base % 1024 == 0,
      "CorrGather requires A1x128/B128x128 native local tiles and 1024-row B alignment");
  systolic_corr::GatherStatistics stats{};
  Submit({}, {local_a, local_b}, [&] {
    c384_initialized_ = false;
    stats = systolic_corr::GatherAt(a, level0, level1, ad.base, bd.base);
  });
  // GatherAt checks magic, busy/error and exactly 1024 committed rows.
  allocator_.Complete(true);
  return stats;
}
void GemminiExecutor::StoreCorrelation(LocalHandle a, LocalHandle b, float* output) {
  const auto& ad = allocator_.Describe(a);
  const auto& bd = allocator_.Describe(b);
  LocalAllocator::Check(ad.rows == 1 && ad.columns == 128 && bd.rows == 128 && bd.columns == 128 &&
      ad.layout == LocalLayout::TileRows && bd.layout == LocalLayout::TileRows,
      "invalid correlation local GEMM operands");
  // LoopMatmul advances its private ACC cursor even for D=0. Reserve all ACC
  // for this external-output wrapper; no undeclared ACC tensor may be pinned.
  const auto acc = Reserve(Acc(0, 128, 128));
  try { Submit({a, b}, {acc}, [&] { systolic_corr::DotAt(ad.base, bd.base, 1024, output); }); }
  catch (...) { Release(acc); throw; }
  WaitExternal();
  Release(acc);
}
void GemminiExecutor::Correlation(const dpvo::CorrelationInputs& in) {
  LocalAllocator::Check(in.a && in.windows && in.dots && in.counters && in.jobs &&
                        dpvo::CorrelationGatherSupported(in), "unsupported local correlation batch");
  GemminiExecutor executor;
  systolic_corr::ProbeAt();
  auto a_desc = Tile(2048, 16, 128); a_desc.rows = 1;
  const auto a = executor.Reserve(a_desc);
  const auto b = executor.Reserve(Tile(0, 128, 128));
  ort_replay::Scope profile("kernel", "correlation.local", "CorrelationBatch", "Gemmini");
  for (size_t j = 0; j < in.jobs; ++j) {
    systolic_corr::Window windows[2];
    for (size_t level = 0; level < 2; ++level) {
      const int64_t* w = in.windows + j * 10 + level * 5;
      windows[level] = {reinterpret_cast<const float*>(static_cast<uintptr_t>(w[0])),
          static_cast<uint32_t>(w[1]), static_cast<uint32_t>(w[2]),
          static_cast<int32_t>(w[3]), static_cast<int32_t>(w[4])};
    }
    const auto stats = executor.Gather(in.a + j * in.channels, windows[0], windows[1], a, b);
    executor.StoreCorrelation(a, b, in.dots + j * 128);
    uint64_t* counters = in.counters + j * 7;
    counters[0] = stats.cycles; counters[1] = stats.useful_bytes;
    counters[2] = stats.strips; counters[3] = stats.rows;
    counters[4] = stats.read_stalls; counters[5] = stats.write_stalls; counters[6] = 1;
  }
  executor.Release(a); executor.Release(b);
  if (profile.Active()) {
    std::ostringstream detail;
    detail << "contract=local_v1;jobs=" << in.jobs << ";b_local_bytes=" << in.jobs * UINT64_C(65536)
           << ";a_read_bytes=" << in.jobs * UINT64_C(512) << ";dot_write_bytes=" << in.jobs * UINT64_C(512)
           << ";b_spill_bytes=0;waits=" << executor.waits_;
    profile.Detail(detail.str().c_str());
  }
}

void GemminiExecutor::SoftAggResidual(const float* x, const int64_t* q,
    size_t batches, size_t edges, size_t channels, size_t groups,
    const float* wg, const float* bg, const float* wf, const float* bf,
    const float* wh, const float* bh, float* output) {
  ort_replay::Scope profile("kernel", "softagg.onchip", "SoftAggResidual", "Gemmini");
  std::vector<size_t> offsets(groups + 1, 0);
  for (size_t e = 0; e < edges; ++e) {
    LocalAllocator::Check(q[e] >= 0 && static_cast<uint64_t>(q[e]) < groups,
                         "SoftAgg C index is outside the Unique inverse range");
    ++offsets[static_cast<size_t>(q[e]) + 1];
  }
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  std::vector<size_t> cursors(offsets.begin(), offsets.end() - 1), edge_ids(edges);
  for (size_t e = 0; e < edges; ++e) edge_ids[cursors[static_cast<size_t>(q[e])]++] = e;
  size_t max_group = 0;
  for (size_t g = 0; g < groups; ++g) max_group = std::max(max_group, offsets[g + 1] - offsets[g]);
  LocalAllocator::Check(max_group <= 128, "SoftAgg C supports complete groups of at most 128 edges");
  const uint32_t tile_rows = max_group <= 32 ? 32 : max_group <= 64 ? 64 : max_group <= 96 ? 96 : 128;
  const uint32_t panel = tile_rows == 32 ? 64 : tile_rows == 64 ? 48 : tile_rows == 96 ? 32 : 16;
  const uint32_t weight_rows = kChannels * panel / 16;
  std::vector<float> y_storage(groups * channels + 16);
  const size_t y_offset = (64 - reinterpret_cast<uintptr_t>(y_storage.data()) % 64) % 64;
  float* const y = y_storage.data() + y_offset / sizeof(float);
  const auto bias_s = Reserve(Rows(3840, 16)), bias_v = Reserve(Rows(3856, 16));
  const auto residual = Reserve(Rows(3872, 1)), out = Reserve(Rows(3873, 1));
  systolic_local::CheckEngine();
  ++traffic_.commands;
  uint64_t engine_before[3] = {};
  if (profile.Active()) for (uint64_t i = 0; i < 3; ++i) engine_before[i] = systolic_local::Status(i + 1);
  Initialize();
  for (size_t b = 0; b < batches; ++b) {
    const float* input = x + b * edges * channels;
    float* result = output + b * edges * channels;
    const auto g_weights = Reserve(Tile(0, kChannels, panel));
    const auto f_weights = Reserve(Tile(weight_rows, kChannels, panel));
    const auto local_x = Reserve(Tile(2 * weight_rows, tile_rows, kChannels));
    const auto s = Reserve(Acc(0, tile_rows, panel)), v = Reserve(Acc(512, tile_rows, panel));
    for (uint32_t c = 0; c < kChannels; c += panel) {
      LoadWeights(wg, c, g_weights); LoadWeights(wf, c, f_weights);
      LoadBias(bg, c, panel, bias_s); LoadBias(bf, c, panel, bias_v);
      for (size_t first = 0; first < groups;) {
        size_t last = first;
        while (last < groups && offsets[last + 1] - offsets[first] <= tile_rows) ++last;
        const uint32_t count = static_cast<uint32_t>(offsets[last] - offsets[first]);
        if (count) {
          LoadIndexed(input, edge_ids, offsets[first], offsets[last], local_x);
          SubmitLocalGemm(local_x, g_weights, s); WaitLocal();
          SubmitLocalGemm(local_x, f_weights, v); WaitLocal();
          SpillGroups(s, v, bias_s, bias_v, out, offsets, first, last, c, y);
        }
        first = last;
      }
    }
    WaitExternal();
    Release(g_weights); Release(f_weights); Release(local_x); Release(s); Release(v);
    // The C1 Y spill is intentional. h consumes it in panels; H stays in ACC.
    constexpr uint32_t h_panel = 128, h_rows = 32;
    const auto h_weights = Reserve(Tile(0, kChannels, h_panel));
    const auto local_y = Reserve(Tile(kChannels * h_panel / 16, h_rows, kChannels));
    const auto h = Reserve(Acc(0, h_rows, h_panel));
    for (uint32_t c = 0; c < kChannels; c += h_panel) {
      LoadWeights(wh, c, h_weights); LoadBias(bh, c, h_panel, bias_s);
      for (size_t first = 0; first < groups; first += h_rows) {
        const uint32_t count = static_cast<uint32_t>(std::min<size_t>(h_rows, groups - first));
        LoadGroups(y, first, count, local_y);
        SubmitLocalGemm(local_y, h_weights, h); WaitLocal();
        StoreResidualRows(h, residual, bias_s, bias_v, out, offsets, edge_ids,
                          first, count, c, input, result);
      }
    }
    WaitExternal(); Release(h_weights); Release(local_y); Release(h);
  }
  Release(bias_s); Release(bias_v); Release(residual); Release(out);
  systolic_local::CheckEngine(); ++traffic_.commands;
  if (profile.Active()) {
    std::ostringstream detail;
    detail << "abi=SAG1;order=panel;B=" << batches << ";E=" << edges << ";G=" << groups
           << ";T=" << tile_rows << ";O=" << panel << ";hT=32;hO=128"
           << ";weight_read_bytes=" << traffic_.weights << ";bias_read_bytes=" << traffic_.bias
           << ";x_read_bytes=" << traffic_.x << ";y_read_bytes=" << traffic_.y_read
           << ";y_write_bytes=" << traffic_.y_write << ";output_write_bytes=" << traffic_.output
           << ";rocc_commands=" << traffic_.commands << ";dma_commands=" << traffic_.dma_commands
           << ";dma_64b_address_spans=" << traffic_.dma_lines
           << ";contract=local_v1;local_gemms=" << local_gemms_
           << ";local_gemm_output_bytes=" << local_gemm_bytes_ << ";sv_h_spill_bytes=0;waits=" << waits_;
    for (uint64_t i = 0; i < 3; ++i)
      detail << (i == 0 ? ";engine_commands=" : i == 1 ? ";engine_read_rows=" : ";engine_write_rows=")
             << systolic_local::Status(i + 1) - engine_before[i];
    profile.Detail(detail.str().c_str());
  }
}
#endif
}  // namespace systolic
}  // namespace onnxruntime

// Retain the external-memory API. Its execution plan now belongs to the EP;
// local GEMM itself never materializes its output to external memory.
void SystolicSoftAggResidual(const float* x, const int64_t* q,
    size_t batches, size_t edges, size_t channels, size_t groups,
    const float* wg, const float* bg, const float* wf, const float* bf,
    const float* wh, const float* bh, float* output) {
  using onnxruntime::systolic::LocalAllocator;
  LocalAllocator::Check(batches > 0 && channels == 384 && groups <= edges,
                        "SoftAgg C expects B>=1, C=384, G<=E");
  constexpr size_t limit = std::numeric_limits<size_t>::max() / sizeof(float);
  LocalAllocator::Check(edges <= limit / channels && (edges == 0 || batches <= limit / (edges * channels)),
                        "SoftAgg C tensor size overflow");
  if (!edges) return;
  LocalAllocator::Check(x && q && wg && bg && wf && bf && wh && bh && output && x != output,
                        "SoftAgg C requires separate input/output and nonnull tensors");
#if defined(__riscv) && defined(SYSTOLIC_FP32) && !defined(SYSTOLIC_FP16)
  onnxruntime::systolic::GemminiExecutor executor;
  executor.SoftAggResidual(x, q, batches, edges, channels, groups, wg, bg, wf, bf, wh, bh, output);
#else
  throw std::runtime_error("SoftAgg C requires a RISC-V FP32 build and SAG1 hardware");
#endif
}
