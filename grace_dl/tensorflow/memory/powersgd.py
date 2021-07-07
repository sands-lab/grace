import tensorflow as tf

from grace_dl.tensorflow import Memory


class PowerSGDMemory(Memory):
    def __init__(self, q_memory, compress_rank=1):
        self.compress_rank = compress_rank
        self.q_memory = q_memory
        self.residuals = {}
        for v in tf.trainable_variables():
            self.residuals[v.name] = tf.Variable(tf.zeros_like(v), trainable=False)
            tensor = v
            n = tensor.get_shape().as_list()[0]
            m = 1
            for dim in tensor.get_shape().as_list()[1:]:
                m = m * dim
            r = int(min([m, n, self.compress_rank]))
            self.q_memory[v.name] = tf.Variable(tf.random.normal([m, r]), trainable=False)

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        tensor_dims = len(tensor.get_shape().as_list())
        if tensor_dims == 1:
            return tensor

        tensor = self.residuals[name] + tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        if ctx is None:
            return []

        tensor_decompressed = compressor.decompress(tensor, ctx)
        op = self.residuals[name].assign(tensor - tensor_decompressed)

        return [op]
