// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace onnxruntime {
namespace systolic {

// FP32 DIM16 local addresses count 64-byte rows, not payload bytes. TileRows
// uses base + ((r/16)*ceil(columns/16)+cb)*16+r%16; ChannelBlocks uses
// base + cb*pitch+r. Rows is a contiguous array of vectors (up to 16 lanes).
enum class LocalMemory { Sp, Acc };
enum class LocalLayout { TileRows, ChannelBlocks, Rows };
enum class LocalAccess : unsigned { Read = 1, Write = 2, ReadWrite = 3 };
struct LocalBuffer {
  LocalMemory memory;
  LocalLayout layout;
  uint32_t base, extent, pitch, rows, columns;
  LocalAccess access = LocalAccess::ReadWrite;
};

class LocalAllocator;
class LocalHandle {
 public:
  LocalHandle() = default;
 private:
  friend class LocalAllocator;
  const LocalAllocator* owner_ = nullptr;
  uint64_t owner_id_ = 0;
  uint64_t generation_ = 0;
  unsigned slot_ = 0;
};

// One execution plan owns this allocator. It never assumes private hardware:
// GemminiExecutor also holds the process-wide MLAS device session. No local
// handle escapes an ORT invocation. Fixed reservations preserve C1 placement;
// Allocate also supports first-fit placement for callers without a fixed base.
class LocalAllocator {
 public:
  LocalAllocator() : identity_(NextIdentity()) {}
  LocalAllocator(const LocalAllocator&) = delete;
  LocalAllocator& operator=(const LocalAllocator&) = delete;

  static void Check(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
  }
  static void Validate(const LocalBuffer& b) {
    Check(b.memory == LocalMemory::Sp || b.memory == LocalMemory::Acc, "invalid local memory kind");
    Check(b.layout == LocalLayout::Rows || b.layout == LocalLayout::TileRows ||
          b.layout == LocalLayout::ChannelBlocks, "invalid local layout");
    const uint32_t capacity = b.memory == LocalMemory::Sp ? 4096 : 1024;
    Check(b.extent && b.base < capacity && b.extent <= capacity - b.base,
          "local allocation exceeds memory capacity");
    Check(b.rows && b.columns && b.columns <= 4096 && b.pitch,
          "invalid local shape/pitch");
    Check(static_cast<unsigned>(b.access) >= 1 && static_cast<unsigned>(b.access) <= 3,
          "invalid local access");
    const uint64_t blocks = (uint64_t(b.columns) + 15) / 16;
    if (b.layout == LocalLayout::Rows) {
      Check(b.columns <= 16 && b.pitch == 1 && b.rows <= b.extent,
            "invalid vector-row layout");
    } else {
      Check(b.base % 16 == 0 && b.extent % 16 == 0 && b.pitch % 16 == 0 &&
                b.rows <= b.pitch && uint64_t(b.pitch) * blocks <= b.extent,
            "invalid tiled extent or allocated row pitch");
      Check((b.layout == LocalLayout::TileRows && b.memory == LocalMemory::Sp) ||
                (b.layout == LocalLayout::ChannelBlocks && b.memory == LocalMemory::Acc),
            "unsupported local memory/layout pair");
    }
  }
  LocalHandle Reserve(const LocalBuffer& b) {
    Validate(b);
    for (const auto& s : slots_) if (s.live && s.buffer.memory == b.memory)
      Check(b.base >= s.buffer.base + s.buffer.extent || s.buffer.base >= b.base + b.extent,
            "overlapping live local allocations");
    for (unsigned i = 0; i < slots_.size(); ++i) if (!slots_[i].live) {
      auto& s = slots_[i];
      s.buffer = b; s.live = true; s.ready = false; s.pending = false;
      ++s.generation;
      LocalHandle h; h.owner_ = this; h.owner_id_ = identity_; h.slot_ = i; h.generation_ = s.generation;
      return h;
    }
    throw std::invalid_argument("too many live local allocations");
  }
  LocalHandle Allocate(LocalBuffer b) {
    const uint32_t capacity = b.memory == LocalMemory::Sp ? 4096 : 1024;
    const uint32_t alignment = b.layout == LocalLayout::Rows ? 1 : 16;
    b.base = 0; Validate(b);
    for (; b.base <= capacity - b.extent; b.base += alignment) {
      bool free = true;
      for (const auto& s : slots_) if (s.live && s.buffer.memory == b.memory &&
          b.base < s.buffer.base + s.buffer.extent && s.buffer.base < b.base + b.extent) free = false;
      if (free) return Reserve(b);
    }
    throw std::invalid_argument("local memory exhausted");
  }
  const LocalBuffer& Describe(LocalHandle h) const { return Get(h).buffer; }
  bool Ready(LocalHandle h) const { return Get(h).ready && !Get(h).pending; }
  void Reshape(LocalHandle h, uint32_t rows, uint32_t columns) {
    auto& s = Get(h);
    Check(!s.pending, "cannot reshape an in-flight local buffer");
    auto b = s.buffer; b.rows = rows; b.columns = columns; Validate(b);
    // Native tile layout depends on columns. A new shape needs a new producer.
    s.buffer = b; s.ready = false;
  }
  void Release(LocalHandle h) {
    auto& s = Get(h);
    Check(!s.pending, "cannot release before local/external completion");
    s.live = false; s.ready = false;
  }
  uint32_t Address(LocalHandle h, uint32_t row, uint32_t block = 0) const {
    const auto& b = Describe(h);
    Check(row < b.rows && block < (b.columns + 15) / 16, "local view exceeds valid shape");
    if (b.layout == LocalLayout::TileRows)
      return b.base + ((row / 16) * ((b.columns + 15) / 16) + block) * 16 + row % 16;
    if (b.layout == LocalLayout::ChannelBlocks) return b.base + block * b.pitch + row;
    return b.base + row;
  }

  // A conservative single outstanding transaction. Complete must follow a real
  // hardware fence/status check, not command acceptance or an elapsed timeout.
  void Begin(std::initializer_list<LocalHandle> reads,
             std::initializer_list<LocalHandle> writes, bool in_place = false) {
    Check(!active_, "local transaction already pending");
    uint32_t r = 0, w = 0;
    pending_count_ = 0;
    for (auto h : reads) {
      const auto& s = Get(h);
      Check(s.ready && !s.pending && (static_cast<unsigned>(s.buffer.access) & 1),
            "local input is unready, failed, or not readable");
      const uint32_t bit = uint32_t(1) << h.slot_;
      if (!(r & bit)) pending_slots_[pending_count_++] = h.slot_;
      r |= bit;
    }
    for (auto h : writes) {
      const auto& s = Get(h);
      Check(!s.pending && (static_cast<unsigned>(s.buffer.access) & 2), "local output is not writable");
      const uint32_t bit = uint32_t(1) << h.slot_;
      if (!((r | w) & bit)) pending_slots_[pending_count_++] = h.slot_;
      w |= bit;
    }
    Check(in_place || !(r & w), "local operation does not support input/output alias");
    Check(r || w, "empty local transaction");
    writes_ = w; active_ = true;
    for (unsigned p = 0; p < pending_count_; ++p) slots_[pending_slots_[p]].pending = true;
  }
  void Complete(bool success) {
    Check(active_, "no local transaction to complete");
    for (unsigned p = 0; p < pending_count_; ++p) {
      const unsigned i = pending_slots_[p];
      auto& s = slots_[i]; s.pending = false;
      if (!success) s.ready = false;
      else if (writes_ & (uint32_t(1) << i)) s.ready = true;
    }
    active_ = false; pending_count_ = writes_ = 0;
  }
  bool Pending() const { return active_; }

 private:
  static uint64_t NextIdentity() {
    static std::atomic<uint64_t> next{0};
    return ++next;
  }
  struct Slot {
    LocalBuffer buffer{};
    uint64_t generation = 0;
    bool live = false, ready = false, pending = false;
  };
  Slot& Get(LocalHandle h) {
    Check(h.owner_ == this && h.owner_id_ == identity_ && h.slot_ < slots_.size(), "foreign local handle");
    auto& s = slots_[h.slot_];
    Check(s.live && s.generation == h.generation_, "released or stale local handle");
    return s;
  }
  const Slot& Get(LocalHandle h) const { return const_cast<LocalAllocator*>(this)->Get(h); }
  std::array<Slot, 32> slots_{};
  const uint64_t identity_;
  std::array<unsigned, 32> pending_slots_{};
  unsigned pending_count_ = 0;
  uint32_t writes_ = 0;
  bool active_ = false;
};

}  // namespace systolic
}  // namespace onnxruntime
