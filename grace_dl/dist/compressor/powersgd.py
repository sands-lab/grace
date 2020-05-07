import torch
from torch import distributed as dist

from grace_dl.dist import Compressor


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

    def __init__(self, rank=1, use_memory=False, world_size=1):
        super().__init__()
        self.world_size = world_size
        self.q_memory = {}
        self.rank = rank
        self.use_memory = use_memory

    def compress(self, tensor, name):
        if tensor.dim() == 1:
            return [tensor], None

        shape = tensor.size()
        matrix = tensor.view([shape[0], -1])
        n, m = matrix.size()
        r = min(n, m, self.rank)
        if self.use_memory and name in self.q_memory:
            q = self.q_memory[name]
        else:
            q = torch.empty(m, r, dtype=matrix.dtype, layout=matrix.layout, device=matrix.device).normal_()
            # q, _ = torch.qr(q)
            orthogonalize(q)

        p = torch.mm(matrix, q)
        dist.all_reduce(p)
        p = p / self.world_size
        # p, _ = torch.qr(p)
        orthogonalize(p)
        q = torch.mm(matrix.t(), p)
        dist.all_reduce(q)
        q = q / self.world_size
        ctx = p, q, shape
        if self.use_memory:
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
