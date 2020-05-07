from torch import distributed as dist

from grace_dl.dist import Communicator


class Allreduce(Communicator):

    def send_receive(self, tensors, name, ctx):
        for tensor_compressed in tensors:
            dist.all_reduce(tensor_compressed)
            if self.compressor.average:
                tensor_compressed.div_(self.world_size)
        return self.compressor.decompress(tensors, ctx)
