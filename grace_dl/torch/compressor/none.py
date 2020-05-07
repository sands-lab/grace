from grace_dl.torch import Compressor


class NoneCompressor(Compressor):
    """Default no-op compression."""

    def compress(self, tensor, name):
        return [tensor], None

    def decompress(self, tensors, ctx):
        tensor, = tensors
        return tensor
