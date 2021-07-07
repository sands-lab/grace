import tensorflow as tf

from grace_dl.tensorflow import Compressor


class ThresholdCompressor(Compressor):

    def __init__(self, threshold):
        super().__init__(tensors_size_are_same=False)
        self.threshold = threshold

    def compress(self, tensor, name):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        thr_mask = tf.math.greater(tf.math.abs(tensor_flatten), self.threshold)
        values = tf.boolean_mask(tensor_flatten, thr_mask)
        indices = tf.reshape(tf.where(thr_mask), [-1])
        ctx = tensor_shape
        values = tf.bitcast(values, tf.int32)
        indices = tf.cast(indices, dtype=tf.int32)
        tensor_compressed = tf.concat([values, indices], 0)
        return [tensor_compressed], ctx

    def decompress(self, tensors_compressed, ctx):
        tensor_compressed, = tensors_compressed
        values, indices = tf.split(tensor_compressed, 2)
        values = tf.bitcast(values, tf.float32)
        tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)
        indices = tf.expand_dims(indices, 1)
        tensor_decompressed = tf.scatter_nd(indices, values, [tensor_size])
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed
