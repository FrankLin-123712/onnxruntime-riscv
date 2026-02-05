#include "cmd_arg.h"
#include "dpvo.h"
#include "utils.h"

#include <iostream>

int main(int argc, char* argv[]) {
  Ort::Env env(dpvo::OrtDebugEnabled() ? ORT_LOGGING_LEVEL_VERBOSE
                                      : ORT_LOGGING_LEVEL_WARNING,
               "dpvo");
  dpvo::RunnerConfig cfg = dpvo::ParseArgs(argc, argv);

  auto images = dpvo::ListImages(cfg.sequence_dir);
  if (images.empty()) {
    std::cerr << "No images found in " << cfg.sequence_dir << "\n";
    return 1;
  }
  auto calib = dpvo::LoadCalibration(cfg.calib_file);

  dpvo::DPVORunner runner(cfg, env);
  runner.ProcessSequence(images, calib);
  runner.Summary();
  return 0;
}
