import torch
from torch.utils.dlpack import to_dlpack

from grace_dl.dist import Compressor


def rdxtopk_cuda(tensor, k):
    import rdxtopk
    # Only applicable to cuda tensors.
    device = tensor.device
    tensor = tensor.cuda()
    indices = torch.arange(tensor.numel(), device=tensor.device).int()
    vals, indices = rdxtopk.topk(tensor, indices, k)
    return vals.to(device), indices.to(device).long()


def cupy_topk(tensor, k):
    import cupy  # conda install -c conda-forge cupy=7.0.0=py37h0c141eb_2
    # Only applicable to cuda tensors. cupy.partition is way much faster than torch.topk for large cuda tensors.
    # Warning!! This is not an exact topk implementation, when there are multiple elements equal to the threshold.
    # For float32 gradients, this rarely happens.
    device = tensor.device
    tensor = tensor.cuda()
    cupy_tensor = cupy.fromDlpack(to_dlpack(tensor))
    threshold = -cupy.partition(-cupy_tensor, k)[k]
    indices, = torch.where(tensor >= threshold.item())
    indices = indices[:k]
    values = tensor[indices]
    return values.to(device), indices.to(device).long()


def sparsify(tensor, compress_ratio, kernel):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    if kernel == 'torch':
        _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    elif kernel =='cupy':
        _, indices = cupy_topk(tensor.abs(), k)
    elif kernel == 'rdxtopk_cuda':
        _, indices = rdxtopk_cuda(tensor.abs(), k)
    values = torch.gather(tensor, 0, indices)
    return values, indices.int()


def desparsify(tensors, numel):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_decompressed.scatter_(0, indices.long(), values)
    return tensor_decompressed


class TopKCompressor(Compressor):

    def __init__(self, compress_ratio, kernel='torch'):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.kernel = kernel

    def compress(self, tensor, name):
        tensors = sparsify(tensor, self.compress_ratio, self.kernel)
        ctx = tensor.size()
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        shape = ctx
        numel = shape.numel()
        tensor_decompressed = desparsify(tensors, numel)
        return tensor_decompressed.view(shape)
