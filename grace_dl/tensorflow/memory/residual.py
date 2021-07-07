import tensorflow as tf

from grace_dl.tensorflow import Memory


class ResidualMemory(Memory):
    def __init__(self, beta=1.0, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma
        for v in tf.trainable_variables():
            self.residuals[v.name] = tf.Variable(tf.zeros_like(v), trainable=False)

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        # if name not in self.residuals:
        #     self.residuals[name] = tf.Variable(tf.zeros_like(tensor), trainable=False)
        # shape_str = ''.join(str(e) for e in tensor.get_shape().as_list())
        # self.residuals[name] = self.residuals_init[shape_str].pop()
        tensor = self.beta * self.residuals[name] + self.gamma * tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed
        update_op = self.residuals[name].assign(residual)
        return [update_op]
