#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

// Low-level local-memory operations. The EP owns allocation, layout validation,
// lifetime and completion; this layer validates physical bounds and emits ISA.
namespace systolic_local {
namespace detail {
struct AccessState {
  std::recursive_mutex mutex;
  bool local_active = false;
  uint64_t generation = 0;
};

// One object across translation units (C++14 inline-function static semantics).
inline AccessState& State() {
  static AccessState state;
  return state;
}
inline bool& ThreadOwnsSession() {
  static thread_local bool owned = false;
  return owned;
}
}  // namespace detail

class Session {
 public:
  Session() : lock_(detail::State().mutex) {
    if (detail::State().local_active)
      throw std::logic_error("Nested Gemmini local session");
    detail::State().local_active = true;
    detail::ThreadOwnsSession() = true;
    ++detail::State().generation;
  }
  // The executor must drain submitted work before releasing this guard,
  // including exceptional exits. A software mutex is not a hardware fence.
  ~Session() {
    detail::ThreadOwnsSession() = false;
    detail::State().local_active = false;
  }
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

 private:
  std::unique_lock<std::recursive_mutex> lock_;
};

class LegacyAccess {
 public:
  explicit LegacyAccess(bool hardware = true)
      : lock_(detail::State().mutex, std::defer_lock) {
    if (!hardware) return;
    lock_.lock();
    if (detail::State().local_active)
      throw std::logic_error("Legacy Gemmini helper would overwrite an active local session");
    ++detail::State().generation;
  }

 private:
  std::unique_lock<std::recursive_mutex> lock_;
};

class LocalAccess {
 public:
  LocalAccess() {
    // Session already owns the device mutex throughout the plan. Checking
    // thread ownership avoids recursively locking for every status/DMA row;
    // another thread cannot use this plan's local operations.
    if (!detail::ThreadOwnsSession())
      throw std::logic_error("Gemmini local operation requires an active session");
  }
};

struct Traffic {
  uint64_t weights = 0, bias = 0, x = 0, y_read = 0, y_write = 0, output = 0;
  uint64_t commands = 0, dma_commands = 0, dma_lines = 0;
  void Dma(const void* address, uint32_t rows, uint32_t columns) {
    ++commands;
    ++dma_commands;
    // Address spans only, not measured cache misses or TileLink requests.
    // Callers use a DRAM row stride divisible by 64 bytes.
    dma_lines += rows * ((reinterpret_cast<uintptr_t>(address) % 64 + columns * 4 + 63) / 64);
  }
};

struct MatmulDescriptor {
  uint32_t a, weights, acc;
  uint32_t rows, columns, depth;
  // Row addresses, not bytes. A advances this pitch per 16-row tile;
  // weights advances its pitch per 16-element K tile; ACC per 16 columns.
  uint32_t a_tile_pitch, weight_tile_pitch, acc_column_pitch;
};

#if defined(__riscv) && defined(SYSTOLIC_FP32) && !defined(SYSTOLIC_FP16)
void Fence();
uint64_t Status(uint64_t selector = 0);
void CheckEngine();
void Initialize(uint32_t dram_row_elements, Traffic& traffic);
void RunGroup(uint32_t s, uint32_t v, uint32_t bias_s, uint32_t bias_v,
              uint32_t count, bool residual, uint32_t output_row, Traffic& traffic);
void LoadWeights(const float* weights, uint32_t base, uint32_t column,
                 uint32_t panel, Traffic& traffic, uint32_t channels = 384,
                 uint32_t local_pitch = 0);
void LoadBias(const float* bias, uint32_t base, uint32_t column,
              uint32_t panel, Traffic& traffic);
void LoadIndexedRows(const float* x, const std::vector<size_t>& edge_ids,
                     size_t begin, size_t end, uint32_t base, Traffic& traffic,
                     uint32_t channels = 384, uint32_t local_pitch = 0);
void LoadGroupRows(const float* y, size_t begin, uint32_t count,
                   uint32_t base, Traffic& traffic, uint32_t channels = 384,
                   uint32_t local_pitch = 0);
void MatmulLocal(const MatmulDescriptor& desc, Traffic& traffic);
// One FP32 DIM16 row. Only DMA counters are updated here; the execution plan
// assigns the semantic traffic category (X, Y spill, or external output).
void LoadRow(const float* source, uint32_t base, Traffic& traffic);
void StoreRow(float* destination, uint32_t base, Traffic& traffic);
#endif
}  // namespace systolic_local
