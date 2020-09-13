import tensorflow as tf

from grace_dl.tensorflow import Communicator
from horovod.tensorflow import allgather


class Allgather(Communicator):

    def __init__(self, compressor, memory, world_size):
        super().__init__(compressor, memory)
        self.world_size = world_size

    def send_receive(self, tensors, ctx):
        tensors_size = []
        tensors_shape = []
        tensors_ag = []
        for tensor in tensors:
            tensors_size.append(tf.reshape(tf.size(tensor), [-1]))
            tensors_shape.append(tf.shape(tensor))
            tensors_1d = tf.reshape(tensor, [-1])
            tensors_ag.append(allgather(tensors_1d))
        tensors_size = tf.concat(tensors_size, 0)

        if self.compressor.tensors_size_are_same:
            tensors_size_list = [tensors_size] * self.world_size
            tensors_size_ag = tf.concat(tensors_size_list, 0)
        else:
            tensors_size_ag = allgather(tensors_size)

        index = [0] * self.world_size
        num = len(tensors)
        decompressed_tensors = []
        for ranki in range(self.world_size):
            tensors_size = tensors_size_ag[num * ranki:num * (ranki + 1)]
            ranki_tensors = []
            for i, (tensor, shape) in enumerate(zip(tensors_ag, tensors_shape)):
                a = index[i]
                b = a + tensors_size[i]
                ranki_tensors.append(tf.reshape(tensor[a:b], shape))
                index[i] = b

            ranki_decompressed = self.compressor.decompress(ranki_tensors, ctx)
            decompressed_tensors.append(ranki_decompressed)

        tensors_aggregated = self.compressor.aggregate(decompressed_tensors)
        return (tensors_aggregated / self.world_size) if self.compressor.average else tensors_aggregated
