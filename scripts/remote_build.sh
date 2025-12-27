#!/bin/bash

TASK_FILE="tasks/build_jetson_cross.yaml"
if [[ $# -gt 0 ]]; then
    TASK_FILE=$1
fi

PROGRESS_ARGS=""
# PROGRESS_ARGS="--info=progress2 --progress"

echo Synchronizing source code
time rsync -a ${PROGRESS_ARGS} ./ g:/home/xiaolonghuang/mllm/ --exclude={"data","perf.*",".git","build"}

time ssh g "source ~/.zshrc; cd /home/xiaolonghuang/mllm/ && CCACHE_SLOPPINESS=time_macros python3 task.py ${TASK_FILE}" | \
    sed -e 's/\/home\/xiaolonghuang\/mllm/\/home\/xiaolong\/mllm/g' \
        -e 's/--sysroot[^ ]* //' \
        -e 's/\/home\/xiaolonghuang\/l4t-gcc\/bin\/aarch64-buildroot-linux-gnu-g++/\/usr\/bin\/g++/'

echo Sync back build artifacts
time rsync -a ${PROGRESS_ARGS}  g:/home/xiaolonghuang/mllm/build/  ./build/ 

rm ./compile_commands.json
sed -i 's/\/home\/xiaolonghuang\/mllm/\/home\/xiaolong\/mllm/g' ./build/compile_commands.json
sed -i 's/--sysroot[^ ]* //' ./build/compile_commands.json
sed -i 's/\/home\/xiaolonghuang\/l4t-gcc\/bin\/aarch64-buildroot-linux-gnu-g++/\/usr\/bin\/g++/' ./build/compile_commands.json

ln -sf ./build/compile_commands.json ./compile_commands.json
