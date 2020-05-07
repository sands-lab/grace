import torch

from grace_dl.torch import Memory


class PowerSGDMemory(Memory):
    def __init__(self, q_memory, compress_rank=1):
        self.compress_rank = compress_rank
        self.q_memory = q_memory
        self.residuals = {}

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if tensor.dim() == 1:
            return tensor

        if name in self.q_memory:
            tensor += self.residuals[name]

        shape = tensor.size()
        n = shape[0]
        m = 1
        for dim in shape[1:]:
            m = m * dim

        r = min(n, m, self.compress_rank)
        normal = torch.empty(m, r, dtype=tensor.dtype, layout=tensor.layout, device=tensor.device).normal_()
        self.q_memory[name] = normal

        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        if ctx is None:
            return

        self.residuals[name] = tensor - compressor.decompress(tensor_compressed, ctx)
