from grace_dl.torch import Communicator
from horovod.torch import allreduce_async_, synchronize


class Allreduce(Communicator):

    def async_send(self, tensors_compressed, name):
        handles = []
        for i, tensor_compressed in enumerate(tensors_compressed):
            handles.append(allreduce_async_(tensor_compressed, self.compressor.average, name + str(i)))
        return handles

    def wait_receive(self, handles, ctx):
        output = [synchronize(h) for h in handles]
        return self.compressor.decompress(output, ctx)
