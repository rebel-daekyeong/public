#!/bin/bash

# Requirement: 2x GPUs.


# Model: meta-llama/Llama-3.2-1B-Instruct
# Query: 1024 input tokens, 6 output tokens, QPS 2/4/6/8, 100 requests
# Resource: 2x GPU
# Approaches:
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex

base_url="https://ray.sw1.rebellions.in"
endpoint="/v1/chat/completions"
result_dir="./results"
model="meta-llama/Llama-3.2-1B-Instruct"
dataset_name="sonnet"
dataset_path="./sonnet.txt"
num_prompts=100
prefix_len=50
input_len=1024
output_len=6

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) base_url="$2" ;;
    --endpoint) endpoint="$2" ;;
    --result-dir) result_dir="$2" ;;
    --model) model="$2" ;;
    --dataset-name) dataset_name="$2" ;;
    --dataset-path) dataset_path="$2" ;;
    --num-prompts) num_prompts="$2" ;;
    --prefix-len) prefix_len="$2" ;;
    --input-len) input_len="$2" ;;
    --output-len) output_len="$2" ;;
    *)
      echo "Invalid argument: $1"
      exit 1 ;;
  esac
  shift 2
done

benchmark() {
  local base_url="$1"
  local qps="$2"
  local tag="$3"

  python3 ./benchmark_serving.py \
          --backend openai-chat \
          --model "$model" \
          --dataset-name "$dataset_name" \
          --dataset-path "$dataset_path" \
          --sonnet-input-len "$input_len" \
          --sonnet-output-len "$output_len" \
          --sonnet-prefix-len "$prefix_len" \
          --num-prompts "$num_prompts" \
          --base-url "$base_url" \
          --endpoint "$endpoint" \
          --save-result \
          --result-dir "$result_dir" \
          --result-filename "$tag"-qps-"$qps".json \
          --request-rate "$qps"
}

visualize() {
  python3 ./visualize_benchmark_results.py
}

main() {
  cd "$(dirname "$0")"

  for method in chunked disagg; do
    for qps in 2 4 6 8; do
      benchmark "$base_url/$method" "$qps" "${method}_prefill"
      sleep 2
    done
  done

  visualize
}

main "$@"
