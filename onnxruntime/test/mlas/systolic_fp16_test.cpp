#include "core/mlas/inc/systolic_fp16.h"
#include "core/mlas/inc/systolic_mlas.h"
#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <chrono>
#include <cfenv>
#include <cmath>

// Independent pre-migration oracle; never linked into production libraries.
namespace reference_half {
inline float Decode(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000) << 16;
  uint32_t e = (h >> 10) & 31, m = h & 1023, bits;
  if (e == 0) {
    if (m == 0) bits = sign;
    else {
      int shift = 0;
      while (!(m & 1024)) { m <<= 1; ++shift; }
      bits = sign | (uint32_t(113 - shift) << 23) | ((m & 1023) << 13);
    }
  } else if (e == 31) bits = sign | 0x7f800000u | (m << 13);
  else bits = sign | ((e + 112) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

inline uint16_t Encode(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint16_t sign = uint16_t((bits >> 16) & 0x8000);
  const uint32_t exp = (bits >> 23) & 255, frac = bits & 0x7fffff;
  if (exp == 255) return sign | (frac ? 0x7e00 : 0x7c00);
  const int e = int(exp) - 127;
  if (e > 15) return sign | 0x7c00;
  if (e < -25) return sign;
  const uint32_t sig = frac | 0x800000;
  const int shift = e < -14 ? -e - 1 : 13;
  uint32_t rounded = sig >> shift;
  const uint32_t rem = sig & ((uint32_t(1) << shift) - 1);
  const uint32_t halfway = uint32_t(1) << (shift - 1);
  rounded += rem > halfway || (rem == halfway && (rounded & 1));
  if (e < -14) return sign | uint16_t(rounded);
  return sign | uint16_t(((e + 14) << 10) + rounded);
}

// Each WS PE rounds a fused multiply-add to half. This is a numerical
// reference primitive, not a claim of cycle/bit parity for a whole tiled GEMM.
inline uint16_t Encode(double f) {
  uint64_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint16_t sign = uint16_t((bits >> 48) & 0x8000);
  const uint64_t exp = (bits >> 52) & 2047, frac = bits & 0xfffffffffffffull;
  if (exp == 2047) return sign | (frac ? 0x7e00 : 0x7c00);
  const int e = int(exp) - 1023;
  if (e > 15) return sign | 0x7c00;
  if (e < -25) return sign;
  const uint64_t sig = frac | (uint64_t(1) << 52);
  const int shift = e < -14 ? 28 - e : 42;
  uint64_t rounded = sig >> shift;
  const uint64_t rem = sig & ((uint64_t(1) << shift) - 1);
  const uint64_t halfway = uint64_t(1) << (shift - 1);
  rounded += rem > halfway || (rem == halfway && (rounded & 1));
  if (e < -14) return sign | uint16_t(rounded);
  return sign | uint16_t(((e + 14) << 10) + rounded);
}

inline uint16_t Fma(uint16_t a, uint16_t b, uint16_t c) {
  // A float intermediate can double-round at a binary16 halfway point.
  return Encode(std::fma(double(Decode(a)), double(Decode(b)), double(Decode(c))));
}
constexpr uint16_t kOne = 0x3c00;
}  // namespace reference_half

namespace {
uint16_t NativeBits(float value) { return systolic_fp16::ToBits(static_cast<_Float16>(value)); }
uint16_t NativeBits(double value) { return systolic_fp16::ToBits(static_cast<_Float16>(value)); }
float NativeFloat(uint16_t bits) { return static_cast<float>(systolic_fp16::FromBits(bits)); }

void CheckBoundary(double value) {
  assert(NativeBits(value) == reference_half::Encode(value));
  const float f = static_cast<float>(value);
  assert(NativeBits(f) == reference_half::Encode(f));
}

void CheckNativeConversions() {
  systolic_fp16::RequireRoundToNearest();
  // Check both signs, halfway values, and adjacent representable float/double
  // inputs. The double neighbours also detect a conversion via float.
  for (unsigned h = 0; h < 0x7bff; ++h) {
    const double midpoint = (double(reference_half::Decode(h)) +
                             double(reference_half::Decode(h + 1))) * 0.5;
    for (double sign : {1.0, -1.0}) {
      const double x = midpoint * sign;
      CheckBoundary(x);
      CheckBoundary(std::nextafter(x, -std::numeric_limits<double>::infinity()));
      CheckBoundary(std::nextafter(x, std::numeric_limits<double>::infinity()));
      const float xf = static_cast<float>(x);
      CheckBoundary(double(std::nextafter(xf, -std::numeric_limits<float>::infinity())));
      CheckBoundary(double(std::nextafter(xf, std::numeric_limits<float>::infinity())));
    }
  }
  for (double x : {0.0, -0.0, 65504.0, -65504.0, 65520.0, -65520.0,
                   1e100, -1e100, 1e-100, -1e-100}) CheckBoundary(x);
  assert(std::isnan(NativeFloat(NativeBits(std::numeric_limits<float>::quiet_NaN()))));
  assert(std::isinf(NativeFloat(NativeBits(std::numeric_limits<float>::infinity()))));
  assert(std::signbit(NativeFloat(NativeBits(-0.0f))));
  // Reject unsupported rounding at the kernel boundary, then restore the caller.
  assert(std::fesetround(FE_DOWNWARD) == 0);
  bool rejected = false;
  try { SystolicHalfMatmul(0,0,0,0,nullptr,0,nullptr,0,nullptr,0,nullptr,0); }
  catch (const std::runtime_error&) { rejected = true; }
  assert(std::fesetround(FE_TONEAREST) == 0);
  assert(rejected);
  // Exercise the actual CPU reference FMA with a halfway product plus a tiny
  // partial sum. The tiny first product would disappear with float rounding.
  const uint16_t a[] = {NativeBits(1.0f), NativeBits(1.5f)};
  const uint16_t b[] = {1, NativeBits(1.0009765625f)};
  uint16_t c = 0;
  SystolicHalfMatmul(0,1,1,2,a,2,b,1,nullptr,1,&c,1);
  assert(c == NativeBits(1.501953125f));
  std::cout << "Native half boundary, direct double conversion, FMA and rounding-mode tests passed\n";
}

void BenchmarkConversions() {
  std::vector<float> values(4096);
  for (size_t i=0; i<values.size(); ++i) values[i] = (float(i)-2048.0f)*0.0137f;
  const volatile float* input = values.data();
  auto measure = [&](bool native) {
    volatile double sum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (size_t i=0; i<262144; ++i) {
      const float x = input[i % values.size()];
      sum += native ? NativeFloat(NativeBits(x)) : reference_half::Decode(reference_half::Encode(x));
    }
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
    return std::make_pair(us, double(sum));
  };
  const auto reference = measure(false), native = measure(true);
  assert(reference.second == native.second);
  std::cout << "Conversion benchmark 262144 float/half/float pairs: reference_us="
            << reference.first << " native_us=" << native.first << " checksum=" << native.second << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  const char mode = argc == 2 && std::string(argv[1]) == "--ws" ? 2 : 0;
  if (argc > 2 || (argc == 2 && mode != 2)) return 2;
  std::cout << "FP16 DIM32 test execution=" << int(mode) << std::endl;
  using namespace systolic_fp16;
  for (unsigned h = 0; h < 65536; ++h) {
    const bool nan = (h & 0x7c00) == 0x7c00 && (h & 1023);
    if (!nan) {
      assert(NativeBits(NativeFloat(uint16_t(h))) == h);
      assert(NativeBits(double(NativeFloat(uint16_t(h)))) == h);
    } else assert(std::isnan(NativeFloat(uint16_t(h))));
  }
  assert(NativeBits(1.0f) == kOne);
  assert(NativeBits(-0.0f) == 0x8000);
  assert(NativeBits(65504.0f) == 0x7bff);
  assert(NativeBits(65520.0f) == 0x7c00);
  assert(NativeBits(std::ldexp(1.0f, -24)) == 1);
  assert(NativeBits(std::ldexp(1.0f, -25)) == 0);
  assert(NativeBits(3 * std::ldexp(1.0f, -25)) == 2);
  assert(NativeBits(2049.0f) == NativeBits(2048.0f));
  assert(NativeBits(2051.0f) == NativeBits(2052.0f));
  // A halfway product plus a very small addend must not double-round via FP32.
  CheckNativeConversions();
  BenchmarkConversions();

  // Non-exact products distinguish FP16 PE accumulation from FP32 FMA.
  // Identical terms avoid depending on the order of the PEs in the array.
  // Read the full accumulator so the small FP32 bias must survive as well.
  {
    const uint16_t v = 0x3c01;  // 1 + 2^-10
    std::vector<uint16_t> a(64, v), b(64, v);
    uint16_t partial = 0;
    for (int i = 0; i < 32; ++i) partial = reference_half::Fma(v, v, partial);
    const float bias = 0.0001f;
    float expected = bias;
    expected += reference_half::Decode(partial);
    expected += reference_half::Decode(partial);
    float actual = 0;
    SystolicHalfMatmul(mode,1,1,64,a.data(),64,b.data(),1,&bias,1,nullptr,1,true,false,1.0f,&actual);
    assert(actual == expected);
    assert(actual != 64 * NativeFloat(v) * NativeFloat(v) + bias);
    std::cout << "FP16 PE rounding and FP32 inter-tile accumulation passed\n";
  }

  // Non-contiguous matrices, odd dimensions, multiple DIM blocks and row bias.
  const size_t m = 3, n = 5, k = 65, lda = 69, ldb = 7, ldc = 8;
  std::vector<uint16_t> a(m*lda, 0), b(k*ldb, 0), c(m*ldc, 0x3555);
  std::vector<float> bias(n, -1.0f);
  for (size_t i=0;i<m;++i) for(size_t t=0;t<k;++t) a[i*lda+t]=NativeBits(float(i+1));
  for (size_t t=0;t<k;++t) for(size_t j=0;j<n;++j) b[t*ldb+j]=NativeBits(float(j+1));
  SystolicHalfMatmul(mode,m,n,k,a.data(),lda,b.data(),ldb,bias.data(),n,c.data(),ldc,true);
  for(size_t i=0;i<m;++i) {
    for(size_t j=0;j<n;++j) assert(NativeFloat(c[i*ldc+j]) == float(k*(i+1)*(j+1)-1));
    for(size_t j=n;j<ldc;++j) assert(c[i*ldc+j] == 0x3555);
  }
  std::vector<float> full(m*ldc, -123.0f);
  SystolicHalfMatmul(mode,m,n,k,a.data(),lda,b.data(),ldb,bias.data(),n,nullptr,ldc,true,false,1.0f,full.data());
  for(size_t i=0;i<m;++i) {
    for(size_t j=0;j<n;++j) assert(full[i*ldc+j] == float(k*(i+1)*(j+1)-1));
    for(size_t j=n;j<ldc;++j) assert(full[i*ldc+j] == -123.0f);
  }
  SystolicHalfMatmul(0,m,n,0,nullptr,0,nullptr,0,bias.data(),n,c.data(),ldc,true,true);
  for(size_t i=0;i<m;++i) for(size_t j=0;j<n;++j) assert(c[i*ldc+j] == 0);
  bool rejected=false;
  try { SystolicHalfMatmul(1,0,0,0,nullptr,0,nullptr,0,nullptr,0,nullptr,0); }
  catch(const std::invalid_argument&) { rejected=true; }
  assert(rejected);
  std::cout << "FP16 conversions (all 65536 encodings), rounding, strides and mode tests passed\n";
}
