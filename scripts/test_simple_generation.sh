#!/bin/bash

/bin/sync
echo 3 | sudo tee /proc/sys/vm/drop_caches 

# Default values
CS=${CS:-32}
MODEL_TYPE=${MODEL_TYPE:-Qwen3-0.6B}
MAX_DECODE_TOKENS=${MAX_DECODE_TOKENS:-32}
IGNORE_EOS=${IGNORE_EOS:-false}
LAZY_LOAD=${LAZY_LOAD:-true}

# Model base directory
MODEL_BASE="/home/xiaolong/llms/${MODEL_TYPE}"

# Input prompt text
PROMPT="The development of artificial intelligence has fundamentally transformed how we approach problem-solving in the modern world. Machine learning, as a subset of AI, represents a paradigm shift from traditional programming methods where developers explicitly code every rule and decision point. Instead, machine learning systems learn patterns and relationships from vast amounts of data, enabling them to make predictions, recognize patterns, and adapt to new situations without being explicitly programmed for each scenario. This approach has proven remarkably effective across numerous domains, from natural language processing and computer vision to recommendation systems and autonomous vehicles. The core principle underlying machine learning is that algorithms can identify complex patterns in data that would be difficult or impossible for humans to manually encode. Neural networks, inspired by the structure of the human brain, consist of interconnected nodes organized in layers that process information through weighted connections. During training, these networks adjust their internal parameters to minimize errors between their predictions and the actual outcomes in the training data. This process, known as backpropagation, allows the network to gradually improve its performance through repeated exposure to examples. Deep learning, which involves networks with many layers, has been particularly successful in handling high-dimensional data such as images, audio, and text. The ability of these systems to automatically extract hierarchical features from raw data has revolutionized fields like medical diagnosis, where AI can now detect diseases from medical images with accuracy rivaling or exceeding human experts. Similarly, language models trained on vast text corpora can generate coherent and contextually appropriate responses, translate between languages, and even assist in creative writing tasks. However, the power of machine learning comes with important considerations regarding data quality, algorithmic bias, computational resources, and the interpretability of model decisions. As these systems become more integrated into critical applications, understanding their limitations and ensuring their responsible deployment becomes increasingly important for both researchers and practitioners in the field."

TRACE_FOLDER="./data/trace/${MODEL_TYPE}-${CS}"
STATE_PATH="./data/state/${MODEL_TYPE}-${CS}"
mkdir -p "${TRACE_FOLDER}"
mkdir -p "${STATE_PATH}"

rm -rf ${STATE_PATH}/qwen3-runner

if [ "$DEBUG" ]; then
  APP="gdb --args ./build/bin/mllm-qwen3-runner"
else
  APP="./build/bin/mllm-qwen3-runner"
fi

# Run generation with input from stdin
echo "${PROMPT}" | ${APP}\
  -m "${MODEL_BASE}/model.mllm" \
  -c "${MODEL_BASE}/config.json" \
  -t "${MODEL_BASE}/tokenizer.json" \
  --state_path "${STATE_PATH}/qwen3-runner" \
  -cs "${CS}" \
  -tf "${TRACE_FOLDER}/1.csv" \
  -mdt "${MAX_DECODE_TOKENS}" \
  $([ "$IGNORE_EOS" = "true" ] && echo "-ie") \
  $([ "$LAZY_LOAD" = "true" ] && echo "-ll")
