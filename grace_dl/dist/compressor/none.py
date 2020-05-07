from grace_dl.dist import Compressor


class NoneCompressor(Compressor):
    """Default no-op compression."""

    def compress(self, tensor, name):
        return [tensor], None

    def decompress(self, tensors, ctx):
        tensor, = tensors
        return tensor
