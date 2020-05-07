import torch

from grace_dl.torch import Compressor


class TernGradCompressor(Compressor):

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()

        sign_gradient = gradient.sign() * scalar
        rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
        sign_gradient[rnd_sample >= abs_gradient] = 0
        new_sign = sign_gradient.sign()  # -1, 0, 1

        tensor_compressed = new_sign.type(torch.int8), scalar.flatten()

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, scalar = tensor_compressed
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape)
