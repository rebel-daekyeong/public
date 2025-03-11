import os
import sys
import torch
import torch.nn as nn

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

# from log_utils import rank_log, get_logger, verify_min_gpu_count

# ---- GPU check ------------
_min_gpu_count = 2

# if not verify_min_gpu_count(min_gpus=_min_gpu_count):
#     print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
#     sys.exit()
# ---------------------------

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed._tensor import Shard, Replicate


"""
This is the script to test Tensor Parallel(TP) on a toy model in a
Megetron-LM SPMD style. We show an E2E working flow from forward,
backward and optimization.

More context about API designs can be found in the design:

https://github.com/pytorch/pytorch/issues/89884.

And it is built on top of Distributed Tensor which is proposed in:

https://github.com/pytorch/pytorch/issues/88838.

We use the example of two `nn.Linear` layers with an element-wise `nn.RELU`
in between to show an example of Megatron-LM, which was proposed in paper:

https://arxiv.org/abs/1909.08053.

The basic idea is that we parallelize the first linear layer by column
and also parallelize the second linear layer by row so that we only need
one all reduce in the end of the second linear layer.

We can speed up the model training by avoiding communications between
two layers.

To parallelize a nn module, we need to specify what parallel style we want
to use and our `parallelize_module` API will parse and parallelize the modules
based on the given `ParallelStyle`. We are using this PyTorch native Tensor
Parallelism APIs in this example to show users how to use them.
"""

class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


"""
Main body of the demo of a basic version of tensor parallel by using
PyTorch native APIs.
"""
# logger = get_logger()

# create a device mesh based on the given world_size.
_world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
init_process_group(backend="nccl")

device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()


print(f"Starting PyTorch TP example on rank {_rank}.")
# assert (
#     _world_size % 2 == 0
# ), f"TP examples require even number of GPUs, but got {_world_size} gpus"

# rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")

# create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
# tp_model = ToyModel().to("cuda")
org_model = ToyModel().to(local_rank)

# Create an optimizer for the parallelized module.
lr = 0.25
optimizer = torch.optim.AdamW(org_model.parameters(), lr=lr, foreach=True)

# Custom parallelization plan for the model
tp_model = parallelize_module(
    module=org_model,
    device_mesh=device_mesh,
    parallelize_plan={
        # "in_proj": ColwiseParallel(),
        # "out_proj": RowwiseParallel(),
        # "in_proj": ColwiseParallel(output_layouts=Shard(1)), # 1: last dim
        # "out_proj": RowwiseParallel(input_layouts=Shard(1)),
        "in_proj": ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
        "out_proj": RowwiseParallel(input_layouts=Shard(1), output_layouts=Shard(0)),
    },
)
# org_model = torch.compile(org_model)
# tp_model = torch.compile(tp_model)

# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
num_iters = 1
# rank_log(_rank, logger, "Tensor Parallel training starting...")

for i in range(num_iters):
    # For TP, input needs to be same across all TP ranks.
    # Setting the random seed is to mimic the behavior of dataloader.
    torch.manual_seed(i)
    # inp = torch.rand(20, 10, device="cuda")
    inp = torch.rand(20, 10, requires_grad=True).to(local_rank)
    output = tp_model(inp)
    # output.sum().backward()
    # optimizer.step()
    # rank_log(_rank, logger, f"Tensor Parallel iter {i} completed")
    org_output = org_model(inp)
    same = "equal" if torch.equal(output, org_output) else "diff"
    print(f"org/tp results are {same} in rank{_rank}")
    almost_same = "almost equal" if torch.allclose(output, org_output) else "almost diff"
    print(f"org/tp results are {almost_same} in rank{_rank}")
    # print(org_output)
    # print(output)
    output.sum().backward()
    optimizer.step()

# rank_log(_rank, logger, "Tensor Parallel training completed!")
destroy_process_group()
