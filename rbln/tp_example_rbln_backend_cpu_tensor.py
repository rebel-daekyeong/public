import os
import sys
import torch
import torch.nn as nn
import rebel

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    loss_parallel,
)

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed._tensor import Shard

from torch._dynamo.backends.common import aot_autograd

# from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch._subclasses.fake_tensor import unset_fake_temporarily

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


def main():
    """
    Main body of the demo of a basic version of tensor parallel by using
    PyTorch native APIs.
    """

    # create a device mesh based on the given world_size.
    _world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)
    # init_process_group(backend="rbln")
    init_process_group(backend="rbln")

    # device_mesh = init_device_mesh(device_type="rbln", mesh_shape=(_world_size,))
    device_mesh = init_device_mesh(device_type="cpu", mesh_shape=(_world_size,))
    _rank = device_mesh.get_rank()

    print(f"Starting PyTorch TP example on rank {_rank}.")
    assert _world_size % 2 == 0, (
        f"TP examples require even number of GPUs, but got {_world_size} gpus"
    )

    # create model and move it to RBLN - init"rbln"_mesh has already mapped RBLN ids.
    model = ToyModel()
    # tp_model = model.to(local_rank)
    # org_model = model.to("rbln")
    org_model = model.to("cpu")

    # Create an optimizer for the parallelized module.
    lr = 0.25
    optimizer = torch.optim.AdamW(org_model.parameters(), lr=lr, foreach=True)

    # Custom parallelization plan for the model
    tp_model = parallelize_module(
        module=org_model,
        device_mesh=device_mesh,
        parallelize_plan={
            "in_proj": ColwiseParallel(),
            "out_proj": RowwiseParallel(),
        },
    )

    def host_compiler(gm, example_inputs):
        gm.print_readable()
        return gm

    def rbln_compiler(gm, example_inputs):
        gm.print_readable()
        print(len(example_inputs))
        # print(example_inputs)
        for ei in example_inputs:
            print(ei.shape)
            print(ei)
        print("-----------------------")
        with unset_fake_temporarily():
            inputs = [torch.zeros(inp.shape, dtype=inp.dtype) for inp in example_inputs]
            # print(inputs)
            scripted_model = torch.jit.trace(gm, inputs)
            print(scripted_model.graph)
            compiled_model = rebel.compile_from_torchscript(scripted_model)
            compiled_model.save("./tp_example.rbln")
            compiled_model.get_num_host_ops()
            compiled_model.get_graph_json()
            runtime = compiled_model.create_runtime(tensor_type="pt")
            return runtime.run
        return gm

    tp_model = torch.compile(
        tp_model,
        backend=aot_autograd(fw_compiler=host_compiler, bw_compiler=host_compiler),
    )

    # print(tp_model)
    # for p in tp_model.parameters():
    #     print(p)ref/examples/distributed/tensor_parallelism/tensor_parallel_example_cpu.pyref/examples/distributed/tensor_parallelism/tensor_parallel_example_cpu.py
    #     print(p.shape)
    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    num_iters = 1

    # with torch.inference_mode():
    for i in range(num_iters):
        # For TP, input needs to be same across all TP ranks.
        # Setting the random seed is to mimic the behavior of dataloader.
        torch.manual_seed(i)
        # inp = torch.rand(20, 10, requires_grad=True).to(local_rank)
        # inp = torch.rand(20, 10, requires_grad=True).to("rbln")
        inp = torch.rand(20, 10, requires_grad=True).to("cpu")
        output = tp_model(inp)
        # output.sum().backward()
        # optimizer.step()
        # with loss_parallel():
        #     loss = output.sum()
        #     loss.backward()
        #     optimizer.step()
        org_output = org_model(inp)
        same = "equal" if torch.equal(output, org_output) else "diff"
        print(f"org/tp results are {same} in rank{_rank}")
        almost_same = (
            "almost equal" if torch.allclose(output, org_output) else "almost diff"
        )
        print(f"org/tp results are {almost_same} in rank{_rank}")
        print(org_output)
        print(output)
        # output.sum().backward()
        # optimizer.step()

    destroy_process_group()


if __name__ == "__main__":
    main()
