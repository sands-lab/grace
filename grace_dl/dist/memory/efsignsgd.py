from grace_dl.torch import Memory


class EFSignSGDMemory(Memory):
    def __init__(self, lr):
        self.residuals = {}
        self.learning_rate = lr

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if name in self.residuals:
            tensor = self.residuals[name] + self.learning_rate * tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed
        self.residuals[name] = residual
