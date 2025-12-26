#!/bin/bash

PROGRESS_ARGS=""
# PROGRESS_ARGS="--info=progress2 --progress"


echo Synchronizing source code
rsync -a ${PROGRESS_ARGS} ./ g:/home/xiaolonghuang/mllm/ --exclude={"data","perf.*",".git","build"}

ssh g 'source ~/.zshrc; cd /home/xiaolonghuang/mllm/ && python3 task.py tasks/build_jetson_cross.yaml' | \
    sed -e 's/\/home\/xiaolonghuang\/mllm/\/home\/xiaolong\/mllm/g' \
        -e 's/--sysroot[^ ]* //' \
        -e 's/\/home\/xiaolonghuang\/l4t-gcc\/bin\/aarch64-buildroot-linux-gnu-g++/\/usr\/bin\/g++/'

echo Sync back build artifacts
rsync -a ${PROGRESS_ARGS}  g:/home/xiaolonghuang/mllm/build/  ./build/ 

rm ./compile_commands.json
sed -i 's/\/home\/xiaolonghuang\/mllm/\/home\/xiaolong\/mllm/g' ./build/compile_commands.json
sed -i 's/--sysroot[^ ]* //' ./build/compile_commands.json
sed -i 's/\/home\/xiaolonghuang\/l4t-gcc\/bin\/aarch64-buildroot-linux-gnu-g++/\/usr\/bin\/g++/' ./build/compile_commands.json

ln -sf ./build/compile_commands.json ./compile_commands.json
