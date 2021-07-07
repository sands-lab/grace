import tensorflow as tf

from grace_dl.tensorflow import Compressor


class DgcCompressor(Compressor):
    """
    (2017). Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training.
    Retrieved from http://arxiv.org/abs/1712.01887
    """
    def __init__(self, compress_ratio):
        super().__init__(tensors_size_are_same=False)
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]

        sample_shape = tf.reshape(tf.convert_to_tensor(max(1, int(elemnum * 0.01)), dtype=tf.int32), [-1])
        sample_index = tf.random.uniform(sample_shape, minval=0, maxval=elemnum, dtype=tf.int32)
        sample_tensor = tf.gather(tensor_flatten, sample_index)

        k = max(1, int(elemnum * self.compress_ratio * 0.01))
        vals, indices = tf.math.top_k(tf.math.abs(sample_tensor), k)
        thr = tf.math.reduce_min(vals)
        mask = tf.math.greater(tf.math.abs(tensor_flatten), thr)

        selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))

        def body(thr, mask, selected):
            thr = tf.cond(selected > 1.25 * max(1, int(elemnum * self.compress_ratio)), lambda: 1.25 * thr,
                          lambda: 0.9 * thr)
            mask = tf.math.greater(tf.math.abs(tensor_flatten), thr)
            selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))
            return thr, mask, selected

        def condition(thr, mask, selected):
            cond_a = selected > 1.25 * max(1, int(elemnum * self.compress_ratio))
            cond_b = selected < 0.8 * max(1, int(elemnum * self.compress_ratio))
            return tf.math.logical_or(cond_a, cond_b)

        thr, mask, new_selected = tf.while_loop(condition, body, (thr, mask, selected), maximum_iterations=20)

        thr = tf.cond(new_selected < 1, lambda: 0.8 * thr, lambda: thr)
        # mask = tf.math.greater_equal(tf.math.abs(tensor_flatten), thr)
        mask = tf.math.greater(tf.math.abs(tensor_flatten), thr)  # fix the dgc NCF data volume issue

        indices = tf.reshape(tf.where(mask), [-1])
        values = tf.gather(tensor_flatten, indices)

        values = tf.bitcast(values, tf.int32)
        indices = tf.cast(indices, dtype=tf.int32)
        tensor_compressed = tf.concat([values, indices], 0)
        ctx = tensor_shape, mask

        return [tensor_compressed], ctx

    def decompress(self, tensors_compressed, ctx):
        tensor_compressed, = tensors_compressed
        tensor_shape, _ = ctx
        values, indices = tf.split(tensor_compressed, 2)
        values = tf.bitcast(values, tf.float32)
        tensor_size = tf.math.reduce_prod(tensor_shape)
        indices = tf.expand_dims(indices, 1)
        tensor_decompressed = tf.scatter_nd(indices, values, [tensor_size])
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed
