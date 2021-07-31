import torch

from grace_dl.dist import Compressor


class ThresholdCompressor(Compressor):

    def __init__(self, threshold):
        super().__init__(tensors_size_are_same=False)
        self.threshold = threshold

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        indices, = torch.where(tensor.abs() >= min(self.threshold, torch.max(tensor)))
        values = tensor[indices]
        ctx = shape
        return [values, indices.int()], ctx

    def decompress(self, tensor_compressed, ctx):
        shape = ctx
        numel = shape.numel()
        values, indices = tensor_compressed
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices.long(), values)
        return tensor_decompressed.view(shape)
