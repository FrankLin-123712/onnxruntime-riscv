// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/systolic/gemmini_local.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/systolic_local.h"
#include "core/mlas/inc/systolic_mlas.h"

namespace onnxruntime {
namespace dpvo { struct CorrelationInputs; }
namespace systolic {

#if defined(__riscv) && defined(SYSTOLIC_FP32) && !defined(SYSTOLIC_FP16)
// A synchronous fused-op execution plan. Device ownership spans every helper
// and every local lease, including CPU metadata work and intentional Y spill.
// There is no cross-node cache and no software-visible hardware token ISA.
class GemminiExecutor {
 public:
  GemminiExecutor();
  ~GemminiExecutor();
  GemminiExecutor(const GemminiExecutor&) = delete;
  GemminiExecutor& operator=(const GemminiExecutor&) = delete;

  LocalHandle Reserve(const LocalBuffer& buffer) { return allocator_.Reserve(buffer); }
  void Release(LocalHandle h) { allocator_.Release(h); }
  void WaitLocal();
  void WaitExternal() { WaitLocal(); }
  void SubmitLocalGemm(LocalHandle a, LocalHandle weights, LocalHandle output);
  systolic_corr::GatherStatistics Gather(const float* a, const systolic_corr::Window& level0,
      const systolic_corr::Window& level1, LocalHandle local_a, LocalHandle local_b);
  void StoreCorrelation(LocalHandle a, LocalHandle b, float* output);

  // Explicit external-memory boundaries for the current C384 execution plan.
  // Initialize before these loaders; none is called implicitly by local GEMM.
  void Initialize() {
    LocalAllocator::Check(!allocator_.Pending(), "cannot reconfigure pending local work");
    systolic_local::Initialize(384, traffic_);
    c384_initialized_ = true;
  }
  void LoadWeights(const float* weights, uint32_t column, LocalHandle dst);
  void LoadBias(const float* bias, uint32_t column, uint32_t panel, LocalHandle dst);
  void LoadIndexed(const float* x, const std::vector<size_t>& ids, size_t begin, size_t end,
                   LocalHandle dst);
  void LoadGroups(const float* y, size_t begin, uint32_t count, LocalHandle dst);
  void LoadResidual(const float* input, LocalHandle dst);
  void Reduce(LocalHandle s, LocalHandle v, LocalHandle bias_s, LocalHandle bias_v,
              LocalHandle output, uint32_t row, uint32_t block, uint32_t count, bool residual);
  void Store(LocalHandle source, float* output);

  static void Correlation(const dpvo::CorrelationInputs& in);

 private:
  friend void ::SystolicSoftAggResidual(const float*, const int64_t*, size_t, size_t, size_t, size_t,
      const float*, const float*, const float*, const float*, const float*, const float*, float*);
  // The external wrapper validates the complete plan before entering here.
  void SoftAggResidual(const float* x, const int64_t* q,
      size_t batches, size_t edges, size_t channels, size_t groups,
      const float* wg, const float* bg, const float* wf, const float* bf,
      const float* wh, const float* bh, float* output);
  void SpillGroups(LocalHandle s, LocalHandle v, LocalHandle bias_s, LocalHandle bias_v,
      LocalHandle output, const std::vector<size_t>& offsets, size_t first, size_t last,
      uint32_t column, float* y);
  void StoreResidualRows(LocalHandle h, LocalHandle residual, LocalHandle bias_s, LocalHandle bias_v,
      LocalHandle output, const std::vector<size_t>& offsets, const std::vector<size_t>& edge_ids,
      size_t first, uint32_t count, uint32_t column, const float* input, float* result);

  template <typename Function>
  void Submit(std::initializer_list<LocalHandle> reads, std::initializer_list<LocalHandle> writes,
              Function function, bool in_place = false) {
    allocator_.Begin(reads, writes, in_place);
    try { function(); }
    catch (...) {
      // Drain before releasing device ownership; partial results stay invalid.
      systolic_local::Fence();
      allocator_.Complete(false);
      throw;
    }
  }
  systolic_local::Session session_;
  LocalAllocator allocator_;
  systolic_local::Traffic traffic_;
  uint64_t waits_ = 0, local_gemms_ = 0, local_gemm_bytes_ = 0;
  bool c384_initialized_ = false;
};
#endif

}  // namespace systolic
}  // namespace onnxruntime
