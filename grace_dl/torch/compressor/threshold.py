import torch

from grace_dl.torch import Compressor


class ThresholdCompressor(Compressor):

    def __init__(self, threshold):
        super().__init__(tensors_size_are_same=False)
        self.threshold = threshold

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        indices, = torch.where(tensor.abs() > self.threshold)
        values = tensor[indices]
        ctx = shape, numel
        return [values, indices], ctx

    def decompress(self, tensor_compressed, ctx):
        shape, numel = ctx
        values, indices = tensor_compressed
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)
