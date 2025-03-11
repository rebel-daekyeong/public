import os
import uvloop
from multiprocessing import freeze_support
from optimum.rbln import RBLNLlamaForCausalLM
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

if __name__ == "__main__":
    freeze_support()

    # Export huggingFace pytorch llama2 model to RBLN compiled model
    model_id = "meta-llama/Llama-3.2-1B"
    if not os.path.exists(model_id):
        compiled_model = RBLNLlamaForCausalLM.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_max_seq_len=4096,
            rbln_tensor_parallel_size=1,
            rbln_batch_size=4,
        )
        compiled_model.save_pretrained(f"./{model_id}")
    else:
        compiled_model = RBLNLlamaForCausalLM.from_pretrained(
            model_id=f"./{model_id}",
            rbln_max_seq_len=4096,
            rbln_tensor_parallel_size=1,
            rbln_batch_size=4,
        )

    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    args.model = f"./{model_id}"
    args.device = "rbln"
    args.max_model_len = 4096
    args.max_num_batched_tokens = 4096
    args.max_num_seqs = 4
    args.block_size = 4096
    args.port = 8001
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
