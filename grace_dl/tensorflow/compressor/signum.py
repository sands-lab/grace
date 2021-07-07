import tensorflow as tf

from grace_dl.tensorflow import Compressor


class SignumCompressor(Compressor):
    """
    (2018). signSGD: Compressed Optimisation for Non-Convex Problems.
    Retrieved from https://arxiv.org/abs/1802.04434
    """

    def __init__(self, momentum):
        super().__init__()
        self.momentum_factor = momentum
        self.momentum = {}
        for v in tf.trainable_variables():
            self.momentum[v.name] = tf.Variable(tf.zeros_like(v), trainable=False)

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = tf.math.add_n(tensors)
        agged_tensor = tf.cast(tf.math.greater_equal(agged_tensor, 0), dtype=tf.float32)
        agged_tensor = agged_tensor * 2.0 - 1.0
        return agged_tensor

    def compress(self, tensor, name):
        """Encoding and compressing the signs """

        # update tensor by momentum
        tensor = (1.0 - self.momentum_factor) * tensor + self.momentum_factor * self.momentum[name]
        op = self.momentum[name].assign(tensor)

        with tf.control_dependencies([op]):
            tensor_shape = tf.shape(tensor)
            tensor_flatten = tf.reshape(tensor, [-1])
            tensor_compressed = tf.math.greater_equal(tensor_flatten, 0)

            return [tensor_compressed], tensor_shape

    def decompress(self, tensors, tensor_shape):
        """Decoding the signs to float format """
        sign_encode, = tensors
        sign_decode = tf.cast(sign_encode, dtype=tf.float32) * 2.0 - 1.0
        tensor_decompressed = tf.reshape(sign_decode, tensor_shape)
        return tensor_decompressed
