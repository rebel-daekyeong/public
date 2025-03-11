import os
import uvloop
from multiprocessing import freeze_support
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser
from vllm.config import KVTransferConfig

#   CUDA_VISIBLE_DEVICES=0 python3 \
#     -m vllm.entrypoints.openai.api_server \
#     --model $model \
#     --port 8100 \
#     --max-model-len 10000 \
#     --gpu-memory-utilization 0.6 \
#     --kv-transfer-config \
#     '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    freeze_support()

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    args.model = "meta-llama/Llama-3.2-1B-Instruct"
    args.port = 8100
    args.max_model_len = 10000
    args.gpu_memory_utilization = 0.6
    args.kv_transfer_config = KVTransferConfig(
        kv_connector="PyNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_buffer_size=5e9,
    )
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
