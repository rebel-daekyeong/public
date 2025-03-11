import asyncio
import logging
import os
import ray
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    OpenAIServingModels,
)

logger = logging.getLogger("ray.serve")


class Worker:
    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.openai_serving_chat: OpenAIServingChat = None
        logger.info(f"Starting with engine args: {self.engine_args}")

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        logger.info(f"Request: {request}")

        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()

            self.openai_serving_chat = OpenAIServingChat(
                engine_client=self.engine,
                model_config=model_config,
                models=OpenAIServingModels(
                    engine_client=self.engine,
                    model_config=model_config,
                    base_model_paths=[
                        BaseModelPath(name=self.model_id, model_path=self.model_id)
                    ],
                    lora_modules=None,
                    prompt_adapters=None,
                ),
                response_role="assistant",
                request_logger=RequestLogger(max_log_len=8192),
                chat_template=None,
                chat_template_content_format="auto",
            )

        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


class DisaggWorker(Worker):
    def __init__(self, kv_transfer_config: KVTransferConfig):
        addresses = Addresses.options(
            name="GlobalAddresses", get_if_exists=True
        ).remote()
        address: str = ray.util.get_node_ip_address()
        addresses.add.remote(rank=kv_transfer_config.kv_rank, address=address)
        master_address: str = ray.get(addresses.get.remote(0))
        kv_transfer_config.kv_ip = master_address

        self.model_id = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
        self.engine_args = AsyncEngineArgs(
            model=self.model_id,
            # max_model_len=10000,
            max_model_len=8192,
            gpu_memory_utilization=0.6,
            kv_transfer_config=kv_transfer_config,
        )
        super().__init__(self.engine_args)


@ray.remote
class Addresses:
    def __init__(self):
        self.addresses: dict[int, str] = {}
        self.ready_events: dict[int, asyncio.Event] = {}

    def add(self, rank: int, address: str) -> None:
        self.addresses[rank] = address
        ready_event = self.ready_events.get(rank, None)
        if ready_event is not None:
            ready_event.set()

    async def get(self, rank: int):
        res = self.addresses.get(rank, None)
        if res is None:
            self.ready_events[rank] = asyncio.Event()
            await self.ready_events[rank].wait()
            del self.ready_events[rank]
            res = self.addresses.get(rank)
        return res
