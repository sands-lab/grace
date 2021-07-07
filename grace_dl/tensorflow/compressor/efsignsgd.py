import tensorflow as tf

from grace_dl.tensorflow import Compressor


class EFSignSGDCompressor(Compressor):
    """
    (2019). Error Feedback Fixes SignSGD and other Gradient Compression Schemes.
    Retrieved from http://arxiv.org/abs/1901.09847
    """
    def __init__(self, lr):
        super().__init__()
        self.learning_rate = lr

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = tf.math.add_n(tensors)
        agged_tensor = agged_tensor / self.learning_rate
        return agged_tensor

    def compress(self, tensor, name):
        """Encoding and compressing the signs """

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        sign_encode = tf.math.greater_equal(tensor_flatten, 0)
        mean = tf.math.reduce_mean(tf.abs(tensor_flatten))
        ctx = tensor_shape
        tensor_compressed = mean, sign_encode

        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        """Decoding the signs to float format """
        mean, sign_encode = tensor_compressed
        tensor_shape = ctx
        sign_decode = tf.cast(sign_encode, dtype=tf.float32) * 2.0 - 1.0
        sign_decode = mean * sign_decode
        tensor_decompressed = tf.reshape(sign_decode, tensor_shape)
        return tensor_decompressed
