import asyncio
import os
import threading
import ray.runtime_context
import torch.nn as nn
from fastapi import FastAPI
import torch
import rebel

from ray import serve
import ray

from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch._dynamo.backends.common import aot_autograd
from torch._subclasses.fake_tensor import unset_fake_temporarily


_world_size: int = int(os.environ.get("WORLD_SIZE", None) or 4)
_world_port: int = int(os.environ.get("WORLD_PORT", None) or 6380)
_num_iter: int = int(os.environ.get("NUM_ITER", None) or 10)


app = FastAPI()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


@ray.remote
class RankCounter:
    def __init__(self, num_counts: int):
        self.count: dict[str, int] = {}
        self.num_counts: int = num_counts

    def increment(self, id="") -> int:
        ret = self.count.get(id, 0)
        self.count[id] = (ret + 1) % self.num_counts
        return ret

    def get_count(self, id="") -> int:
        return self.count.get(id, 0)


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


@ray.remote(num_cpus=1, resources={"ATOM": 1})
class DModel:
    def __init__(self, world_size: int, rank: int, world_port: int):
        self.addresses = Addresses.options(
            name="GlobalAddresses", get_if_exists=True
        ).remote()

        gpu_ids = ray.get_gpu_ids()
        self.local_rank = int(gpu_ids[0]) if len(gpu_ids) > 0 else 0
        self.world_size = world_size
        self.rank = rank
        torch.rbln.set_device(self.local_rank)

        address: str = ray.util.get_node_ip_address()
        self.addresses.add.remote(rank=self.rank, address=address)
        master_address: str = ray.get(self.addresses.get.remote(0))

        print(f"Available resources: {ray.available_resources()}")
        print(f"GPU IDs: {gpu_ids}")
        print(f"World size: {self.world_size}")
        print(f"Rank: {self.rank}, Local rank: {int(self.local_rank)}")
        print(f"Current device: {torch.rbln.current_device()}")
        print(f"Node IP: {address}")
        print(f"Master IP: {master_address}")

        self.pg: torch.distributed.ProcessGroup = init_process_group(
            backend="rbln",
            init_method=f"tcp://{master_address}:{world_port}",
            world_size=self.world_size,
            rank=self.rank,
            device_id=torch.device(f"rbln:{self.local_rank}"),
        )

        self.device_mesh: torch.distributed.device_mesh.DeviceMesh = init_device_mesh(
            device_type="cpu", mesh_shape=(self.world_size,)
        )
        assert self.device_mesh.get_rank() == self.rank

        self.org_model = ToyModel().to("cpu")

        self.optimizer = torch.optim.AdamW(
            self.org_model.parameters(), lr=0.25, foreach=True
        )

        # self.tp_model = self.org_model
        self.tp_model = parallelize_module(
            module=self.org_model,
            device_mesh=self.device_mesh,
            parallelize_plan={
                # "in_proj": ColwiseParallel(),
                # "out_proj": RowwiseParallel(),
                # "in_proj": ColwiseParallel(output_layouts=Shard(1)), # 1: last dim
                # "out_proj": RowwiseParallel(input_layouts=Shard(1)),
                "in_proj": ColwiseParallel(
                    input_layouts=Replicate(), output_layouts=Shard(1)
                ),
                "out_proj": RowwiseParallel(
                    input_layouts=Shard(1), output_layouts=Shard(0)
                ),
            },
        )

        # self.org_model = torch.compile(self.org_model)
        # self.tp_model = torch.compile(self.tp_model)
        def host_compiler(gm, example_inputs):
            gm.print_readable()
            return gm

        def rbln_compiler(gm, example_inputs):
            gm.print_readable()
            with unset_fake_temporarily():
                inputs = [
                    torch.zeros(inp.shape, dtype=inp.dtype) for inp in example_inputs
                ]
                scripted_model = torch.jit.trace(gm, inputs)
                print(scripted_model.graph)
                compiled_model = rebel.compile_from_torchscript(scripted_model)
                compiled_model.save("./tp_example.rbln")
                compiled_model.get_num_host_ops()
                compiled_model.get_graph_json()
                runtime = compiled_model.create_runtime(tensor_type="pt")
                return runtime.run
            return gm

        self.tp_model = torch.compile(
            self.tp_model,
            backend=aot_autograd(fw_compiler=host_compiler, bw_compiler=host_compiler),
        )

        for i in range(_num_iter):
            torch.manual_seed(i)
            inp = torch.rand(20, 10, requires_grad=True).to("cpu")
            output = self.tp_model(inp)
            org_output = self.org_model(inp)
            same = "equal" if torch.equal(output, org_output) else "diff"
            print(f"org/tp results are {same} in rank{self.rank}")
            almost_same = (
                "almost equal" if torch.allclose(output, org_output) else "almost diff"
            )
            print(f"org/tp results are {almost_same} in rank{self.rank}")
            # output.sum().backward()
            # self.optimizer.step()

        self.sync()
        print(f"rank{self.rank} thread id: {threading.get_ident()}")

    def get(self, input: torch.Tensor) -> AsyncCollectiveTensor:
        # output: AsyncCollectiveTensor = self.tp_model(input.to(self.local_rank))
        # output.wait()
        output = self.tp_model(input.to(self.local_rank))

        print(output)
        return output.to(input.device)

    def delete(self):
        destroy_process_group(self.pg)

    def sync(self):
        barrier(self.pg)


@serve.deployment(name="Gateway", max_ongoing_requests=1, max_queued_requests=1)
@serve.ingress(app)
class Gateway:
    def __init__(self, world_size: int, world_port: int):
        self.models = []
        for rank in range(0, world_size):
            self.models.append(
                DModel.options(
                    name=f"DModel-{rank}",
                    placement_group_capture_child_tasks=False,
                    get_if_exists=True,
                ).remote(world_size, rank, world_port)
            )
        self.world_size = world_size

        print("Wating for models to be ready...")
        ray.wait([model.sync.remote() for model in self.models])
        print("All models are ready!")

    @app.get("/")
    def get(self):
        if len(self.models) == 0:
            return "No models available"

        input = torch.rand(20, 10)

        futures = []
        for rank in range(0, _world_size):
            futures.append(self.models[rank].get.remote(input))

        ouptut = torch.Tensor()
        for rank in range(0, _world_size):
            ouptut = torch.concat((ray.get(futures[rank]), ouptut), dim=0)

        print(ouptut)
        return str(ouptut.tolist())

    @app.delete("/")
    def delete(self):
        for model in self.models:
            ray.get(model.delete.remote())
        self.models.clear()


model = Gateway.bind(_world_size, _world_port)
