from grace_dl.torch import Communicator
from horovod.torch import broadcast_async, synchronize


class Broadcast(Communicator):

    def __init__(self, compressor, memory, world_size):
        super().__init__(compressor, memory)
        self.world_size = world_size

    def async_send(self, tensors_compressed, name):
        handles = []
        for root_rank in range(self.world_size):
            rank_handles = []
            for i, tensor_compressed in enumerate(tensors_compressed):
                rank_handles.append(broadcast_async(tensor_compressed, root_rank, name + str(root_rank) + '_' + str(i)))
            handles.append(rank_handles)
        return handles

    def wait_receive(self, handles, ctx):
        tensors_decompressed = []
        for ranki in handles:
            tensors_compressed = [synchronize(h) for h in ranki]
            tensor_decompressed = self.compressor.decompress(tensors_compressed, ctx)
            tensors_decompressed.append(tensor_decompressed)
        tensor_aggregated = self.compressor.aggregate(tensors_decompressed)
        return (tensor_aggregated / self.world_size) if self.compressor.average else tensor_aggregated
