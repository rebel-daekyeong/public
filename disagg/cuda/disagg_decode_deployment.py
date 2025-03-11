import os
from disagg.cuda.disagg_deployment import DisaggWorker
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request
from vllm.config import KVTransferConfig
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
import logging


logger = logging.getLogger("ray.serve")

decode_app = FastAPI()


@serve.deployment
@serve.ingress(decode_app)
class DecodeWorker(DisaggWorker):
    def __init__(self, world_size: int, rank: int):
        super().__init__(
            KVTransferConfig(
                kv_connector=os.environ.get("KV_CONNECTOR", "PyNcclConnector"),
                kv_role="kv_consumer",
                kv_rank=rank,
                kv_parallel_size=world_size,
                kv_buffer_size=5e9,
            )
        )

    @decode_app.get("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        return await super().create_chat_completion(request, raw_request)


decode_model = DecodeWorker.bind(2, 1)
