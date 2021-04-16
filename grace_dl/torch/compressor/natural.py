import torch
from grace_dl.torch import Compressor
from torch.utils.cpp_extension import load
cnat = load(
    'cnat', ['cnat_cuda.cpp', 'cnat_cuda_kernel.cu'])

class NaturalCompressor(Compressor):
    def __init__(self):
        super().__init__()

    def compress(self, tensor, name):
        shape = tensor.size()
        return [cnat.compress(tensor.flatten())], shape


    def decompress(self, tensor_compressed, shape):
        tensor_compressed, = tensor_compressed
        return cnat.decompress(tensor_compressed).view(shape)
