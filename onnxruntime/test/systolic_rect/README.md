# Rectangular FP32 LoopConv tests

Run from the ONNX Runtime repository root. No Python packages or model downloads
are required for the standalone test. The header encoder uses the rectangular
LoopConv CONFIG_1..6 ABI in this workspace, not the older square-only ABI.

## Host: decode the emitted command stream and compare with scalar convolution

```sh
g++ -std=c++14 -O2 -Wall -Wextra -Werror -I onnxruntime \
  onnxruntime/test/systolic_rect/conv_rect_test.cpp -o /tmp/conv_rect_host
/tmp/conv_rect_host
```

Optional memory checking: replace `-O2` with
`-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`.
LeakSanitizer requires an environment without ptrace restrictions.

The 18 cases cover batch=2, square/rectangular input, 1/3/7 kernels, stride 1/2,
odd/even dimensions, spatial/channel tails, bias, zero initialization, ReLU,
multiple channel partial sums, and repeated calls. Inputs are binary fractions
so scalar and accelerated sums can be compared at `1e-4` without conflating
normal floating-point reordering with address errors. Stage 2 also checks all
22 DPVO shapes against capacity constraints and stage-1 command counts.

## Spike: execute the actual RoCC instructions

Set `RECT_TOOLS` to the existing RISC-V toolchain installation, and put `dtc` in
PATH (in this workspace: `chipyard/.conda-env/bin`).

```sh
"${RECT_TOOLS}/bin/riscv64-unknown-linux-gnu-g++" \
  -std=c++14 -O2 -static -march=rv64imafdc -mabi=lp64d \
  -DSYSTOLIC_FP32 -DRECT_RISCV -DFOR_FIRESIM -I onnxruntime \
  onnxruntime/test/systolic_rect/conv_rect_test.cpp \
  onnxruntime/core/mlas/lib/systolic/systolic.cpp -o /tmp/conv_rect_spike
LD_LIBRARY_PATH="${RECT_TOOLS}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${RECT_TOOLS}/bin/spike" --extension=gemmini \
  "${RECT_TOOLS}/riscv64-unknown-elf/bin/pk" /tmp/conv_rect_spike
```

Here `FOR_FIRESIM` only suppresses diagnostic printing in the standalone MLAS
source; this test has no runner `mlockall` call. Build the full Patchify runner
without `--for_firesim` for Spike. Spike checks arithmetic and command ABI, not
RTL timing, queue hazards, or FPGA speedup. See dpvo_runner/docs/patchify_optimization.md
for full-model validation and the staged FireSim procedure.
