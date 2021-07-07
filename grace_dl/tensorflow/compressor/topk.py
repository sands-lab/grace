import tensorflow as tf

from grace_dl.tensorflow import Compressor


def sparsify(tensor, compress_ratio):
    elemnum = tensor.get_shape().as_list()[0]
    k = max(1, int(elemnum * compress_ratio))
    _, indices = tf.math.top_k(tf.math.abs(tensor), k, sorted=False)
    values = tf.gather(tensor, indices)
    return values, indices


def desparsify(indices, values, tensor_size):
    indices = tf.expand_dims(indices, 1)
    tensor = tf.scatter_nd(indices, values, [tensor_size])
    return tensor


class TopKCompressor(Compressor):
    def __init__(self, compress_ratio):
        super().__init__()
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]

        values, indices = sparsify(tensor_flatten, self.compress_ratio)

        indices = tf.cast(indices, tf.int32)
        values = tf.bitcast(values, tf.int32)
        tensor_compressed = tf.concat([values, indices], 0)
        ctx = tensor_shape, elemnum
        return [tensor_compressed], ctx

    def decompress(self, tensors_compressed, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        tensor_compressed, = tensors_compressed
        values, indices = tf.split(tensor_compressed, 2)
        values = tf.bitcast(values, tf.float32)
        tensor_shape, tensor_size = ctx
        #tensor_size = tf.math.reduce_prod(tensor_shape)

        tensor_decompressed = desparsify(indices, values, tensor_size)

        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed
