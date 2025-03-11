import logging
import os
from disagg.cuda.disagg_deployment import Worker
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest


logger = logging.getLogger("ray.serve")

chunked_app = FastAPI()


@serve.deployment
@serve.ingress(chunked_app)
class ChunkedWorker(Worker):
    def __init__(self):
        self.model_id = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
        self.engine_args = AsyncEngineArgs(
            model=self.model_id,
            # max_model_len=10000,
            max_model_len=8192,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.6,
        )
        super().__init__(self.engine_args)

    @chunked_app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        return await super().create_chat_completion(request, raw_request)


chunked_model = ChunkedWorker.bind()
