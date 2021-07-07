import tensorflow as tf

from grace_dl.tensorflow import Compressor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""

    def compress(self, tensor, name):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor

        if tensor.dtype.is_floating:
            # Only allow compression from other floating point types
            tensor_compressed = tf.cast(tensor, dtype=tf.float16)
        return [tensor_compressed], tensor.dtype

    def decompress(self, tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""

        tensor_decompressed, = tensor
        dtype = ctx
        if dtype.is_floating:
            tensor_decompressed = tf.cast(tensor, dtype=dtype)
        return tensor_decompressed
