import torch

from grace_dl.dist import Compressor


class NaturalCompressor(Compressor):

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        sign = torch.sign(tensor)
        tensor_abs = tensor.abs()
        tensor_log = torch.log2(tensor_abs)
        log_ceil = tensor_log.ceil().char()
        log_floor = tensor_log.floor().char()
        q = 2 ** log_floor
        p = ((2 ** log_ceil) - tensor_abs) / q
        p[q == 0] = 0.5
        exponent = log_floor + torch.bernoulli(1 - p)

        exponent = exponent.clamp(-110, 17) - 18
        exponent[sign == -1] += 127
        exponent[sign == 0] = -128

        return exponent, shape

    def decompress(self, exponent, shape):
        sign = exponent >= 0
        exponent[sign] -= 127
        exponent += 18
        tensor_abs = 2 ** exponent.float()
        tensor = torch.where(sign, -tensor_abs, tensor_abs)
        return tensor.view(shape)
