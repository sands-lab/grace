import tensorflow as tf

from grace_dl.tensorflow import Compressor


class AdapSparseCompressor(Compressor):
    """
    Gradient Sparsification for Communication-Efficient Distributed Optimization.
    J. Wangni, J. Wang, J. Liu, and T. Zhang. In NeurIPS, 2018."""

    def __init__(self, compress_ratio):
        super().__init__(tensors_size_are_same=False)
        self.k = compress_ratio

    def compress(self, tensor):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        tensor_size = tf.cast(tf.size(tensor), dtype=tensor.dtype)

        prob = self.k * tensor_size * tf.abs(tensor_flatten) / tf.math.reduce_sum(tf.abs(tensor_flatten))
        prob = tf.minimum(prob, 1.0)

        c = tf.constant(2.0)

        def body(c, prob):
            mask = tf.less(prob, 1.0)
            size_indices = tf.cast(tf.size(tf.where(mask)), dtype=tf.float32)
            sum_prob = tf.math.reduce_sum(tf.boolean_mask(prob, mask))
            c = ((self.k - 1.0) * tensor_size + size_indices) / sum_prob
            prob = tf.minimum(c * prob, 1.0)
            return c, prob

        def condition(c, prob):
            return tf.greater(c, 1.0)

        res = tf.while_loop(condition, body, (c, prob))
        prob = res[1]

        rnd_sample = tf.random.uniform(tf.shape(tensor_flatten))
        mask = tf.less(rnd_sample, prob)
        indices = tf.reshape(tf.where(mask), [-1])
        values = tf.gather(tensor_flatten / prob, indices)
        values = tf.bitcast(values, tf.int32)
        indices = tf.cast(indices, dtype=tf.int32)
        tensor_compressed = tf.concat([values, indices], 0)
        return [tensor_compressed], tensor_shape

    def decompress(self, tensors_compressed, tensor_shape):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        tensor_compressed, = tensors_compressed
        values, indices = tf.split(tensor_compressed, 2)
        values = tf.bitcast(values, tf.float32)
        tensor_size = tf.math.reduce_prod(tensor_shape)
        indices = tf.expand_dims(indices, 1)
        tensor_decompressed = tf.scatter_nd(indices, values, [tensor_size])
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed
