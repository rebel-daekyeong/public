import copy
import logging
import os

from aiohttp import ClientSession, ClientTimeout
from fastapi import FastAPI
from quart import make_response
from ray import serve
from starlette.requests import Request
from vllm.entrypoints.openai.protocol import ChatCompletionRequest


logger = logging.getLogger("ray.serve")

proxy_app = FastAPI()


@serve.deployment
@serve.ingress(proxy_app)
class ProxyWorker:
    def __init__(self):
        logger.info("Starting proxy.")

    async def forward_request(self, url: str, data):
        async with ClientSession(timeout=ClientTimeout(total=6 * 60 * 60)) as session:
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
            async with session.get(url=url, json=data, headers=headers) as response:
                if response.status == 200:
                    # if response.headers.get('Transfer-Encoding') == 'chunked':
                    if True:
                        async for chunk_bytes in response.content.iter_chunked(1024):
                            yield chunk_bytes
                    else:
                        content = await response.read()
                        yield content

    @proxy_app.post("/v1/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        logger.info(f"Request: {request}")

        original_request_data = await raw_request.json()

        prefill_request = copy.deepcopy(original_request_data)
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1

        # finish prefill
        async for _ in self.forward_request(
            "http://disagg-ray-service-serve-svc:8000/prefill/v1/completions",
            prefill_request,
        ):
            continue

        # return decode
        generator = self.forward_request(
            "http://disagg-ray-service-serve-svc:8000/decode/v1/completions",
            original_request_data,
        )
        response = await make_response(generator)
        response.timeout = None

        return response


proxy_model = ProxyWorker.bind()
