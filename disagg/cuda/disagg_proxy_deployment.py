import copy
import logging
import os

from aiohttp import ClientSession, ClientTimeout
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request
from starlette.responses import StreamingResponse
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

    @proxy_app.post("/v1/chat/completions")
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
            "http://disagg-ray-service-serve-svc:8000/disagg/prefill/v1/chat/completions",
            prefill_request,
        ):
            continue

        # return decode
        generator = self.forward_request(
            "http://disagg-ray-service-serve-svc:8000/disagg/decode/v1/chat/completions",
            original_request_data,
        )

        return StreamingResponse(
            content=generator,
            media_type="text/event-stream" if request.stream else "application/json",
        )


proxy_model = ProxyWorker.bind()


#  curl -X POST https://ray.sw1.rebellions.in/disagg/v1/chat/completions \
# -H "Content-Type: application/json" \
# -d '{
#   "model": "meta-llama/Llama-3.2-1B-Instruct",
#   "messages": [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is the capital of France?"}
#   ],
#   "max_tokens": 50,
#   "temperature": 0.7,
#   "stream": false
# }'
