from grace_dl.tensorflow import Compressor


class NoneCompressor(Compressor):
    """Default no-op compression."""

    def compress(self, tensor, name):
        """Returns the tensor unmodified."""
        return [tensor], None

    def decompress(self, tensors, ctx):
        """Returns the tensor unmodified."""
        tensor, = tensors
        return tensor
