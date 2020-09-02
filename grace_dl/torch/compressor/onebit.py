import torch

from grace_dl.torch import Compressor


class OneBitCompressor(Compressor):

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        mask0 = tensor < 0
        sum0 = torch.sum(tensor[mask0])
        num0 = torch.sum(mask0).float()
        mean0 = sum0 / num0 if num0 > 0 else sum0

        mask1 = ~mask0
        sum1 = torch.sum(tensor[mask1])
        num1 = numel - num0
        mean1 = sum1 / num1 if num1 > 0 else sum1

        tensor_compressed = mask0.type(torch.uint8), mean0, mean1

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        mask0, mean0, mean1 = tensor_compressed
        mask0= mask0.bool()
        tensor_decompressed = mask0 * mean0 + ~mask0 * mean1
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed
