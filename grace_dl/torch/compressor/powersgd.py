import torch

from grace_dl.torch import Compressor
from horovod.torch import allreduce_


@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


class PowerSGDCompressor(Compressor):
    """
    PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization.
    T. Vogels, S. P. Karimireddy, and M. Jaggi. In NeurIPS, 2019.
    """

    def __init__(self):
        super().__init__()
        self.q_memory = {}

    def compress(self, tensor, name):
        if tensor.dim() == 1:
            return [tensor], None

        shape = tensor.size()
        matrix = tensor.view([shape[0], -1])
        q = self.q_memory[name]
        # q, _ = torch.qr(q)
        orthogonalize(q)

        p = torch.mm(matrix, q)
        p = allreduce_(p)
        # p, _ = torch.qr(p)
        orthogonalize(p)
        q = torch.mm(matrix.t(), p)
        q = allreduce_(q)
        ctx = p, q, shape
        self.q_memory[name] = q
        return [], ctx

    def decompress(self, tensors, ctx):
        if ctx is None:
            tensor, = tensors
            return tensor
        p, q, tensor_shape = ctx
        new_tensor = torch.mm(p, q.t())
        tensor_decompressed = new_tensor.view(tensor_shape)
        return tensor_decompressed
