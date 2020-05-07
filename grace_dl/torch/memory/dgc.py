import torch

from grace_dl.torch import Memory
from horovod.torch import allreduce_


class DgcMemory(Memory):
    def __init__(self, momentum, gradient_clipping):
        self.gradient_clipping = gradient_clipping
        self.momentum = momentum
        self.gradients = {}
        self.residuals = {}

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if self.gradient_clipping:
            tensor_squ_sum = torch.sum(tensor * tensor)
            clipping_val = torch.sqrt(allreduce_(tensor_squ_sum, average=True, name=name))
            tensor = tensor.clamp(-clipping_val, clipping_val)
        if name in self.residuals:
            self.residuals[name] = self.momentum * self.residuals[name] + tensor
        else:
            self.residuals[name] = tensor
        if name in self.gradients:
            self.gradients[name] += self.residuals[name]
            tensor = self.gradients[name]
        else:
            self.gradients[name] = tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        shape, mask, _ = ctx
        not_mask = ~mask.view(shape)
        temp = self.residuals[name] * not_mask
        self.residuals[name] = temp
        temp = self.gradients[name] * not_mask
        self.gradients[name] = temp
