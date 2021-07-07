import torch
from grace_dl.torch import Compressor
from torch.utils.cpp_extension import load
import cnat_cuda
# cnat = load(
#     'cnat', ['/data/scratch/hang/grace/grace_dl/dist/compressor/cnat_cuda.cpp', '/data/scratch/hang/grace/grace_dl/dist/compressor/cnat_cuda_kernel.cu'])


class NaturalCompressor_CUDA(Compressor):
    def __init__(self):
        super().__init__()

    def compress(self, tensor, name):
        device = tensor.device
        tensor = tensor.cuda()
        shape = tensor.size()
        tensor_compressed = cnat_cuda.compress(tensor.flatten())
        return [tensor_compressed.to(device)], shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, = tensor_compressed
        device = tensor_compressed.device
        tensor_decompressed = cnat_cuda.decompress(tensor_compressed.cuda())
        return tensor_decompressed.to(device).view(shape)
