// Optional synchronous replay instrumentation for statically linked runners.
// No dependency on a runner, hardware counters, or ORT's wall-clock profiler.
#pragma once
#include <cstdint>

#if defined(__GNUC__) && !defined(_WIN32)
extern "C" {
bool ort_replay_profile_enabled(const char* kind) __attribute__((weak));
uint64_t ort_replay_profile_begin(const char* kind, const char* name,
                                  const char* op, const char* provider) __attribute__((weak));
void ort_replay_profile_end(uint64_t id) __attribute__((weak));
void ort_replay_profile_detail(uint64_t id, const char* text) __attribute__((weak));
}
#endif

namespace ort_replay {
inline bool Enabled(const char* kind) {
#if defined(__GNUC__) && !defined(_WIN32)
  return ort_replay_profile_enabled && ort_replay_profile_begin &&
         ort_replay_profile_end && ort_replay_profile_detail &&
         ort_replay_profile_enabled(kind);
#else
  (void)kind;
  return false;
#endif
}

class Scope {
 public:
  Scope(const char* kind, const char* name, const char* op = "", const char* provider = "") {
#if defined(__GNUC__) && !defined(_WIN32)
    if (Enabled(kind)) id_ = ort_replay_profile_begin(kind, name, op, provider);
#endif
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() { End(); }
  bool Active() const { return id_ != 0; }
  void Detail(const char* text) const {
#if defined(__GNUC__) && !defined(_WIN32)
    if (id_) ort_replay_profile_detail(id_, text);
#endif
  }
  void End() {
#if defined(__GNUC__) && !defined(_WIN32)
    if (id_) ort_replay_profile_end(id_);
#endif
    id_ = 0;
  }
 private:
  uint64_t id_ = 0;
};
}  // namespace ort_replay
