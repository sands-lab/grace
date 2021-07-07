import tensorflow as tf

from grace_dl.tensorflow import Compressor


class AdaqCompressor(Compressor):
    """
    (2017). Communication quantization for data-parallel training of deep neural networks.
    https://doi.org/10.1109/MLHPC.2016.4
    """
    def __init__(self, compress_ratio):
        super().__init__(tensors_size_are_same=False)
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        def quan(tensor, tensor_mask, compress_ratio):
            tensor_value = tf.boolean_mask(tensor, tensor_mask)
            mask_size = tf.reduce_sum(tf.cast(tensor_mask, dtype=tf.int32))
            sample_size = tf.cast(tf.reshape((tf.math.ceil(tf.cast(mask_size, dtype=tf.float32) * 0.01)), [-1]),
                                  dtype=tf.int32)
            sample_index = tf.random.uniform(sample_size, minval=0, maxval=mask_size, dtype=tf.int32)
            sample_tensor = tf.gather(tensor_value, sample_index)

            k = tf.cast((tf.math.ceil(tf.cast(mask_size, dtype=tf.float32) * 0.01 * compress_ratio)),
                        dtype=tf.int32)
            vals, indices = tf.math.top_k(tf.math.abs(sample_tensor), k)
            thr = tf.math.reduce_min(vals)
            tensor_masked = tf.cast(tensor_mask, dtype=tf.float32) * tensor
            mask = tf.math.greater(tf.math.abs(tensor_masked), thr)

            # fix the issue of sampling in topk
            selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))
            elemnum = tf.cast(mask_size, dtype=tf.float32)

            def body(thr, mask, selected):
                thr = tf.cond(selected > 1.25 * tf.math.ceil(elemnum * compress_ratio), lambda: 1.25 * thr,
                              lambda: 0.9 * thr)
                mask = tf.math.greater_equal(tf.math.abs(tensor_masked), thr)
                selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))
                return thr, mask, selected

            def condition(thr, mask, selected):
                cond_a = selected > 1.25 * tf.math.ceil(elemnum * compress_ratio)
                cond_b = selected < 0.8 * tf.math.ceil(elemnum * compress_ratio)
                return tf.math.logical_or(cond_a, cond_b)

            thr2, mask2, selected2 = tf.while_loop(condition, body, (thr, mask, selected), maximum_iterations=20)
            thr2 = tf.cond(selected2 < 1, lambda: 0.8 * thr2, lambda: thr2)
            mask2 = tf.math.greater(tf.math.abs(tensor_masked), thr2)

            indices = tf.reshape(tf.where(mask2), [-1])
            tensor_value2 = tf.boolean_mask(tensor_masked, mask2)
            mean = tf.reshape(tf.math.reduce_mean(tensor_value2), [-1])

            return indices, mean, mask2

        tensor_shape = tf.shape(tensor)
        tensor_size = tf.size(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        plus_mask = tf.math.greater(tensor_flatten, 0)
        minus_mask = tf.math.less(tensor_flatten, 0)
        plus_indices, plus_mean, plus_mask = quan(tensor_flatten, plus_mask, self.compress_ratio)
        minus_indices, minus_mean, minus_mask = quan(tensor_flatten, minus_mask, self.compress_ratio)

        plus_mean = tf.bitcast(plus_mean, tf.int32)
        plus_indices = tf.reshape(tf.cast(plus_indices, dtype=tf.int32), [-1])
        minus_mean = tf.bitcast(minus_mean, tf.int32)
        minus_indices = tf.reshape(tf.cast(minus_indices, dtype=tf.int32), [-1])
        plus_indices_size = tf.reshape(tf.size(plus_indices), [-1])
        tensor_compressed = tf.concat([plus_mean, minus_mean, plus_indices_size, plus_indices, minus_indices], 0)
        ctx = tensor_shape, tensor_size
        return [tensor_compressed], ctx

    def decompress(self, tensors_compressed, ctx):
        tensor_compressed, = tensors_compressed
        plus_mean = tensor_compressed[0]
        minus_mean = tensor_compressed[1]
        plus_indices_size = tensor_compressed[2]
        plus_indices = tensor_compressed[3:3 + plus_indices_size]
        minus_indices = tensor_compressed[3 + plus_indices_size:]

        plus_mean = tf.bitcast(plus_mean, tf.float32)
        minus_mean = tf.bitcast(minus_mean, tf.float32)
        tensor_shape, tensor_size = ctx

        plus_mean = tf.ones(tf.shape(plus_indices), dtype=tf.float32) * plus_mean
        minus_mean = tf.ones(tf.shape(minus_indices), dtype=tf.float32) * minus_mean
        indices = tf.expand_dims(plus_indices, 1)
        plus_means = tf.scatter_nd(indices, plus_mean, [tensor_size])
        indices = tf.expand_dims(minus_indices, 1)
        tensor_decompressed = tf.tensor_scatter_nd_update(plus_means, indices, minus_mean)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed
