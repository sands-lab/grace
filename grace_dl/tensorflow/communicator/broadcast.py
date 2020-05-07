from grace_dl.tensorflow import Communicator
from horovod.tensorflow import broadcast


class Broadcast(Communicator):

    def __init__(self, compressor, memory, world_size):
        super().__init__(compressor, memory)
        self.world_size = world_size

    def send_receive(self, tensors, ctx):
        decompressed_tensors = []
        for ranki in range(self.world_size):
            ranki_tensors = [broadcast(tensor, root_rank=ranki) for tensor in tensors]
            ranki_decompressed = self.compressor.decompress(ranki_tensors, ctx)
            decompressed_tensors.append(ranki_decompressed)

        aggregated_tensor = self.compressor.aggregate(decompressed_tensors)
        return aggregated_tensor
