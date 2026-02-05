#!/bin/bash
release_path="Debug"

for var in "$@"; do
    if [ "$var" = "--config=Release" ]; then
        release_path="Release"
    fi
done

echo "Building against ${release_path}"
root_path=../../
build_path=${root_path}/build/${release_path}

extra_libs=""
extra_defs=""
extra_providers=""
training_libs=""

for var in "$@"; do
    if [ "$var" = "--use_hwacha" ]; then
        echo "Building with hwacha support"
        extra_defs="-DUSE_HWACHA ${extra_defs}"
        extra_providers="${build_path}/libonnxruntime_providers_hwacha.a ${extra_providers}"
    fi
    if [ "$var" = "--for_firesim" ]; then
        echo "Building with mlockall for running on Firesim"
        extra_defs="-DFOR_FIRESIM ${extra_defs}"
    fi
    if [ "$var" = "--enable_training" ]; then
        extra_libs="${build_path}/tensorboard/libtensorboard.a ${extra_libs}"
        training_libs="${build_path}/libonnxruntime_training_runner.a ${build_path}/libonnxruntime_training.a"
    fi
    if [ "$var" = "--ort_debug" ]; then
        echo "Building with DPVO_ORT_DEBUG enabled"
        extra_defs="-DDPVO_ORT_DEBUG=1 ${extra_defs}"
    fi
done

rm -f dpvo_runner
make -s -j16 dpvo_runner root_path="${root_path}" build_path="${build_path}" extra_libs="${extra_libs}" \
                       extra_defs="${extra_defs}" training_libs="${training_libs}" extra_providers="${extra_providers}"
echo "Done. dpvo_runner built (custom dpvo::scatter_max CPU kernel registered; corr/BA still CPU placeholders)."
