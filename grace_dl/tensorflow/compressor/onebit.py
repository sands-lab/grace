import tensorflow as tf

from grace_dl.tensorflow import Compressor


class OneBitCompressor(Compressor):
    """
    (2014). 1-bit stochastic gradient descent and its application to data-parallel distributed training of speech DNNs.
    Retrieved from https://www.microsoft.com/en-us/research/publication/1-bit-stochastic-gradient-descent-and-application-to-data-parallel-distributed-training-of-speech-dnns/
    """
    def compress(self, tensor, name):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])

        mask0 = tf.math.less(tensor_flatten, 0)
        sum0 = tf.math.reduce_sum(tf.boolean_mask(tensor_flatten, mask0))
        num0 = tf.math.reduce_sum(tf.cast(mask0, dtype=tf.float32))
        num0 = tf.where(tf.math.greater(num0, 0), num0, 1.0)
        mean0 = sum0 / num0

        mask1 = tf.math.logical_not(mask0)
        sum1 = tf.math.reduce_sum(tf.boolean_mask(tensor_flatten, mask1))
        num1 = tf.math.reduce_sum(tf.cast(mask1, dtype=tf.float32))
        num1 = tf.where(tf.math.greater(num1, 0), num1, 1.0)
        mean1 = sum1 / num1

        mean0 = tf.reshape(mean0, [-1])
        mean1 = tf.reshape(mean1, [-1])
        mean = tf.concat([mean0, mean1], 0)
        ctx = tensor_shape
        tensor_compressed = mask0, mean

        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        tensor_shape = ctx
        mask0, mean = tensor_compressed
        mean0, mean1 = tf.split(mean, 2)
        mask0 = tf.cast(mask0, dtype=tf.float32)
        tensor_decompressed = mask0 * mean0 + (1 - mask0) * mean1
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed
