# Implementing GRACE components

## Compressor
The `Compressor` class has two methods, `compress` and `decompress`. As described in the [paper](http://hdl.handle.net/10754/662495), there are many ways to
implement a compressor. According to the categorization of each compressor, additional utility functions can be used
such as `quantize`, `dequantize`, `sparisify`, `desparsify`, `pack`, and `unpack`. Examples of these functions are
implemented for each compressor and can be found bundled with each compressor.
- `compress` is the compression operator. It expects a single tensor as input, and returns a collection of tensors along
with a context. These values should be passed to `decompress` as is.
- `decompress` takes the output of `compress` and returns a single tensor.

A developer should keep in mind that only the collection of tensors is communicated across workers. The context is kept
locally for each worker.

Generally it is expected, but not enforced, that the result of `decompress(**compress(tensor))` has the same shape and
data type as `tensor`. A good compressor would try to keep these values very close, but also try to reduce the size of
the collection of tensors in comparison with the original tensor.

## Memory
The `Memory` class has two methods, `update` and `compensate`. As described in the [paper](http://hdl.handle.net/10754/662495), a typical residual memory
implementation is provided by `ResidualMemory`.
- The `update` method would store the residual difference between the node' original tensor and the decompression of the
compressed original tensor (i.e. `residual[name] = tensor - decompress(compress(tensor))`). Generally, an `update` method
expects the original tensor, the name that uniquely identifies it, the compressor (used for decompression), and
the output of `compressor.compress(tensor)`. This method has no return value, with an exception for TensorFlow 1.15,
in which case some `Variable.assign` operations may be needed.
- The `compensate` in this case has to apply the residual memory to the next iteration over this tensor. This is
achieved by adding the new tensor with the residual memory stored for the tensor name which is used to uniquely identify
each tensor (i.e. `new_tensor = residula[name] + tensor`). To make `ResidualMemory` a bit more general, optional
parameters can be provided during construction. Generally, a `compensate` method expects the original tensor, and
the name that uniquely identifies it. This method has returns the compensated tensor.

Other compression-specific implementations of `Memory` can be found in the same directory. There is no specific reason
not to bundle a compression-specific `Memory` implementation with the appropriate `Compressor` implementation.

## Communicator
Three implementations of `Communicator` classes are provided for each framework, each with a different collective method.
A `Communicator` implementation needs to override at least the `send_receive` method, or in the case of
Horovod 0.18.2 PyTorch which splits this step to the `async_send` and `wait_receive` methods. These methods expect
a collection of tensors, the context which is the result of `compression.compress` method, and the name of the
compressed tensor.
It has to be noted that not all communicator are compatible with every compressor. A notable example is the `Allreduce`
communicator which can be used with `RandomKCompressor` but not with `TopKCompressor`.

## Putting it all together
```python
# Horovod PyTorch
from grace_dl.torch import Communicator, Compressor, Memory

class MyCommunicator(Communicator):
    ...


class MyCompressor(Compressor):
    ...

class MyMemory(Memory):
    ...

grc = MyCommunicator(MyCompressor(...), MyMemory(...), ...)
```
