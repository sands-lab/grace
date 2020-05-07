from grace_dl.tensorflow import Communicator
from horovod.tensorflow import _allreduce


class Allreduce(Communicator):
    """
    A horovod.tensrflow implementation of Communicator for grace framework.
    """

    def __init__(self, compressor, memory, world_size):
        super().__init__(compressor, memory)
        self.world_size = world_size

    def send_receive(self, tensors_compressed, ctx):
        compressed_tensors = []
        for i, tensor_compressed in enumerate(tensors_compressed):
            summed_tensor = _allreduce(tensor_compressed)
            reduced = (summed_tensor / self.world_size) if self.compressor.average else summed_tensor
            compressed_tensors.append(reduced)
        return self.compressor.decompress(compressed_tensors, ctx)
