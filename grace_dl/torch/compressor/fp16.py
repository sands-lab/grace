import torch

from grace_dl.torch import Compressor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""

    def compress(self, tensor, name):
        """Downcasts the tensor to 16-bit."""
        dtype = tensor.dtype
        if dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor = tensor.type(torch.float16)
        return [tensor], dtype

    def decompress(self, tensors, dtype):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed, = tensors
        if dtype.is_floating_point:
            tensor_decompressed = tensor_decompressed.type(dtype)
        return tensor_decompressed
