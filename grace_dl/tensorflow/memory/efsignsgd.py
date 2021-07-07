import tensorflow as tf

from grace_dl.tensorflow import Memory


class EFSignSGDMemory(Memory):

    def __init__(self, lr):
        self.residuals = {}
        self.learning_rate = lr
        for v in tf.trainable_variables():
            self.residuals[v.name] = tf.Variable(tf.zeros_like(v), trainable=False)

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        tensor = self.residuals[name] + self.learning_rate * tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed
        update_op = self.residuals[name].assign(residual)
        return [update_op]
