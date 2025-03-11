import asyncio
import shutil
import uuid
import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, Request, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, conint, HttpUrl
from pathvalidate import sanitize_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Benchmark WebUI")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

for dir in [UPLOAD_DIR, RESULTS_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class BenchmarkConfig(BaseModel):
    input_len: conint(gt=0) = 1024
    output_len: conint(gt=0) = 6
    prefix_len: conint(ge=0) = 50


class BenchmarkRequest(BenchmarkConfig):
    base_url: HttpUrl = "https://ray.sw1.rebellions.in"
    dataset_name: str
    dataset_type: str = "sonnet"
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    num_prompts: conint(gt=0) = 100


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)


manager = ConnectionManager()


# 전역 상태
class BenchmarkState:
    def __init__(self):
        self.active_process: Optional[asyncio.subprocess.Process] = None
        self.output_buffer: list = []
        self.lock = asyncio.Lock()


benchmark_state = BenchmarkState()


# 미들웨어
@app.middleware("http")
async def global_error_handler(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(f"Global error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )


# 라우트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "status": "success"}
    )


@app.post("/upload/")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        safe_filename = sanitize_filename(file.filename)
        if not safe_filename:
            raise ValueError("Invalid filename")

        file_location = UPLOAD_DIR / safe_filename

        async with benchmark_state.lock:
            with open(file_location, "wb") as buffer:
                contents = await file.read()
                buffer.write(contents)

        return {"filename": safe_filename, "status": "success"}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File upload failed: {str(e)}",
        )


@app.get("/datasets/")
async def list_datasets():
    try:
        files = [f.name for f in UPLOAD_DIR.glob("*") if f.is_file()]
        return {"datasets": files, "status": "success"}
    except Exception as e:
        logger.error(f"Dataset list error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list datasets",
        )


@app.post("/run-benchmark/")
async def run_benchmark(request: BenchmarkRequest):
    global benchmark_state

    async with benchmark_state.lock:
        if (
            benchmark_state.active_process
            and not benchmark_state.active_process.returncode
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Benchmark already running",
            )

        # verify file path
        safe_filename = sanitize_filename(request.dataset_name)
        dataset_path = UPLOAD_DIR / safe_filename
        if not dataset_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Dataset file not found"
            )

        cmd = [
            "bash",
            BASE_DIR / "disagg_performance_benchmark.sh",
            "--model",
            request.model_name,
            "--dataset-name",
            request.dataset_type,
            "--dataset-path",
            str(dataset_path),
            "--input-len",
            str(request.input_len),
            "--output-len",
            str(request.output_len),
            "--prefix-len",
            str(request.prefix_len),
            "--num-prompts",
            str(request.num_prompts),
            "--base-url",
            request.base_url.unicode_string().strip("/"),
            "--endpoint",
            "/v1/chat/completions",
            "--result-dir",
            str(RESULTS_DIR),
        ]

        try:
            benchmark_state.output_buffer = []
            benchmark_state.active_process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )

            asyncio.create_task(stream_output())
            asyncio.create_task(monitor_process())

            return {"status": "success", "message": "Benchmark started"}
        except Exception as e:
            logger.error(f"Benchmark start error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start benchmark",
            )


async def stream_output():
    while (
        benchmark_state.active_process
        and not benchmark_state.active_process.stdout.at_eof()
    ):
        data = await benchmark_state.active_process.stdout.read(1024)
        if not data:
            break
        message = data.decode().strip()
        benchmark_state.output_buffer.append(message)
        await broadcast_to_clients(message)


async def monitor_process():
    await benchmark_state.active_process.wait()
    if benchmark_state.active_process.returncode == 0:
        benchmark_state.active_process = None
    await broadcast_to_clients("BENCHMARK_COMPLETED")


async def broadcast_to_clients(message: str):
    for client_id in manager.active_connections.copy():
        try:
            await manager.send_message(client_id, message)
        except Exception as e:
            logger.error(f"Client {client_id} send error: {str(e)}")
            manager.disconnect(client_id)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)
    try:
        for message in benchmark_state.output_buffer:
            await websocket.send_text(message)

        while True:
            await websocket.receive_text()
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        manager.disconnect(client_id)


@app.get("/visualizations")
async def get_visualizations():
    images = [f.name for f in RESULTS_DIR.glob("*.png")]
    return {"visualizations": images, "status": "success"}


@app.get("/visualization/{image_name}")
async def get_visualization(image_name: str):
    safe_name = sanitize_filename(image_name)
    image_path = RESULTS_DIR / safe_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


@app.delete("/clean-results/")
async def clean_results():
    try:
        async with benchmark_state.lock:
            if benchmark_state.active_process:
                benchmark_state.active_process.terminate()

            for dir in [RESULTS_DIR]:
                shutil.rmtree(dir, ignore_errors=True)
                dir.mkdir(parents=True, exist_ok=True)

            benchmark_state.output_buffer = []
            return {"visualizations": [], "status": "success"}
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Cleanup failed")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
