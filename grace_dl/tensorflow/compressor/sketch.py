import tensorflow as tf

from grace_dl.tensorflow import Compressor


class SketchCompressor(Compressor):
    """
    SketchML: Accelerating Distributed Machine Learning with Data Sketches.
    J. Jiang, F. Fu, T. Yang, and B. Cui. SIGMOD, 2018.
    """

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def compress(self, tensor, name):
        import tensorflow_probability as tfp
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])

        x = tensor_flatten
        edges = tfp.stats.quantiles(x, num_quantiles=self.quantiles, interpolation='linear')
        bins = tf.cast(tfp.stats.find_bins(x, edges), dtype=tf.int32)
        means = tf.math.unsorted_segment_mean(x, bins, num_segments=self.quantiles)

        tensor_compressed = tf.cast(bins, dtype=tf.uint8 if self.quantiles < 256 else tf.uint16)
        means = tf.reshape(means, [-1])
        ctx = tensor_shape
        tensor_compressed = tensor_compressed, means

        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        tensor_shape = ctx
        tensor_compressed, means = tensor_compressed
        bins = tf.cast(tensor_compressed, dtype=tf.int32)
        tensor_decompressed = tf.gather(means, bins)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed
