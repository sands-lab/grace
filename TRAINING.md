# Training with GRACE

## Horovod

The GRACE framework has been seamlessly integrated with Horovod 0.18.2. After following the
[Installation guide](INSTALLING.md) the only modifications needed for a training script that uses Horovod's
`DistributedOptimizer` or `DistributedGradientTape` is to replace the optional compression parameter with
the mandatory `grace` as follows:

```python
# Horovod TensorFlow
import horovod.tensorflow as hvd
from grace_dl.tensorflow.communicator.allgather import Allgather
from grace_dl.tensorflow.compressor.topk import TopKCompressor
from grace_dl.tensorflow.memory.residual import ResidualMemory

world_size = hvd.size()
grc = Allgather(TopKCompressor(0.3), ResidualMemory(), world_size)

tape = hvd.DistributedGradientTape(tape, grace=grc)
# or
optimizer = hvd.DistributedOptimizer(optimizer, grace=grc)

# or with helper
from grace_dl.tensorflow.helper import grace_from_params
params = {'compressor': 'topk', 'memory': 'residual', 'communicator': 'allgather'}
grc = grace_from_params(params)
``` 
```python
# Horovod PyTorch
import horovod.torch as hvd
from grace_dl.torch.communicator.allgather import Allgather
from grace_dl.torch.compressor.topk import TopKCompressor
from grace_dl.torch.memory.residual import ResidualMemory

world_size = hvd.size()
grc = Allgather(TopKCompressor(0.3), ResidualMemory(), world_size)

optimizer = hvd.DistributedOptimizer(optimizer, grace=grc, named_parameters=model.named_parameters())

# or with helper
from grace_dl.torch.helper import grace_from_params
params = {'compressor': 'topk', 'memory': 'residual', 'communicator': 'allgather'}
grc = grace_from_params(params)
``` 

[Here](examples) you can find some simple examples adapted from the examples provided by Horovod.

## PyTorch distributed (DDP)
The PyTorch distributed implementation follows closely the Horovod PyTorch implementation, replacing the communication
collectives with those of `torch.distributed`.
```python
# PyTorch distributed
from grace_dl.dist.communicator.allgather import Allgather
from grace_dl.dist.compressor.topk import TopKCompressor
from grace_dl.dist.memory.residual import ResidualMemory

grc = Allgather(TopKCompressor(0.3), ResidualMemory())

# or with helper
from grace_dl.dist.helper import grace_from_params
params = {'compressor': 'topk', 'memory': 'residual', 'communicator': 'allgather'}
grc = grace_from_params(params)
``` 
