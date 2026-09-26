// Native IEEE binary16 values and bit-preserving Gemmini/ORT boundaries.
#pragma once
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <stdexcept>

#if !defined(__FLT16_MANT_DIG__) || __FLT16_MANT_DIG__ != 11 || __FLT16_MAX_EXP__ != 16
#error This build requires compiler support for IEEE binary16 _Float16
#endif
#if defined(__riscv) && defined(SYSTOLIC_FP16) && !defined(__riscv_zfh)
#error RISC-V FP16 builds require Zfh; compile with -march=rv64imafdc_zfh
#endif

namespace systolic_fp16 {
static_assert(sizeof(_Float16) == sizeof(uint16_t), "binary16 must occupy two bytes");

// These functions copy representations; they do not convert numbers to integers.
inline _Float16 FromBits(uint16_t bits) {
  _Float16 value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline uint16_t ToBits(_Float16 value) {
  uint16_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Native conversions use the floating-point environment. Do not silently
// change a caller's rounding mode: this inference path requires nearest-even.
inline void RequireRoundToNearest() {
  if (std::fegetround() != FE_TONEAREST)
    throw std::runtime_error("FP16 inference requires FE_TONEAREST rounding");
}

constexpr uint16_t kOne = 0x3c00;
}  // namespace systolic_fp16
