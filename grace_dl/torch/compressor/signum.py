import torch

from grace_dl.torch import Compressor


class SignumCompressor(Compressor):

    def __init__(self, momentum):
        super().__init__(average=False)
        self.momentum = momentum
        self.momentums = {}

    def compress(self, tensor, name):
        """Encoding and compressing the signs """
        shape = tensor.size()
        tensor = tensor.flatten()

        # update tensor by momentum
        if name in self.momentums:
            tensor = (1.0 - self.momentum) * tensor + self.momentum * self.momentums[name]
        self.momentums[name] = tensor
        tensor_compressed = tensor >= 0
        return [tensor_compressed.type(torch.uint8)], shape

    def decompress(self, tensors, shape):
        sign_encode, = tensors
        """Decoding the signs to float format """
        sign_decode = sign_encode.type(torch.float32) * 2 - 1
        tensor_decompressed = sign_decode.view(shape)
        return tensor_decompressed

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = sum(tensors)
        agged_tensor = agged_tensor >= 0
        agged_tensor = agged_tensor * 2.0 - 1.0
        return agged_tensor
