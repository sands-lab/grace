import torch

from grace_dl.torch import Compressor


class EFSignSGDCompressor(Compressor):

    def __init__(self, lr):
        super().__init__(average=False)
        self.learning_rate = lr

    def compress(self, tensor, name):
        """Encoding and compressing the signs """
        shape = tensor.size()
        tensor = tensor.flatten()

        sign_encode = tensor >= 0
        mean = tensor.abs().mean()
        tensor_compressed = mean, sign_encode.type(torch.uint8)

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        """Decoding the signs to float format """
        mean, sign_encode = tensor_compressed
        sign_decode = sign_encode.type(torch.float32) * 2 - 1
        sign_decode = mean * sign_decode
        tensor_decompressed = sign_decode.view(shape)
        return tensor_decompressed

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors) / self.learning_rate
