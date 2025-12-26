#!/bin/bash

echo Synchronizing source code
rsync -a --info=progress2 --progress ./ g:/home/xiaolonghuang/mllm/ --exclude={"data","perf.*",".git","build"}
