def grace_from_params(params):
    import horovod.torch as hvd
    world_size = hvd.size()
    comp = params.get('compressor', 'none')
    mem = params.get('memory', 'none')
    comm = params.get('communicator', 'allreduce')
    if comp == 'dgc':
        from grace_dl.torch.compressor.dgc import DgcCompressor
        compressor = DgcCompressor(compress_ratio=0.01)
    elif comp == 'efsignsgd':
        from grace_dl.torch.compressor.efsignsgd import EFSignSGDCompressor
        compressor = EFSignSGDCompressor(lr=0.1)
    elif comp == 'fp16':
        from grace_dl.torch.compressor.fp16 import FP16Compressor
        compressor = FP16Compressor()
    elif comp == 'natural':
        from grace_dl.torch.compressor.natural import NaturalCompressor
        compressor = NaturalCompressor()
    elif comp == 'none':
        from grace_dl.torch.compressor.none import NoneCompressor
        compressor = NoneCompressor()
    elif comp == 'onebit':
        from grace_dl.torch.compressor.onebit import OneBitCompressor
        compressor = OneBitCompressor()
    elif comp == 'powersgd':
        from grace_dl.torch.compressor.powersgd import PowerSGDCompressor
        compressor = PowerSGDCompressor()
    elif comp == 'qsgd':
        from grace_dl.torch.compressor.qsgd import QSGDCompressor
        compressor = QSGDCompressor(quantum_num=64)
    elif comp == 'randomk':
        from grace_dl.torch.compressor.randomk import RandomKCompressor
        compressor = RandomKCompressor(compress_ratio=0.01)
    elif comp == 'signsgd':
        from grace_dl.torch.compressor.signsgd import SignSGDCompressor
        compressor = SignSGDCompressor()
    elif comp == 'signum':
        from grace_dl.torch.compressor.signum import SignumCompressor
        compressor = SignumCompressor(momentum=0.9)
    elif comp == 'terngrad':
        from grace_dl.torch.compressor.terngrad import TernGradCompressor
        compressor = TernGradCompressor()
    elif comp == 'threshold':
        from grace_dl.torch.compressor.threshold import ThresholdCompressor
        compressor = ThresholdCompressor(threshold=0.01)
    elif comp == 'topk':
        from grace_dl.torch.compressor.topk import TopKCompressor
        compressor = TopKCompressor(compress_ratio=0.01)
    else:
        raise NotImplementedError(comp)

    if mem == 'dgc':
        from grace_dl.torch.memory.dgc import DgcMemory
        memory = DgcMemory(momentum=0.9, gradient_clipping=False)
    elif mem == 'none':
        from grace_dl.torch.memory.none import NoneMemory
        memory = NoneMemory()
    elif mem == 'powersgd':
        from grace_dl.torch.memory.powersgd import PowerSGDMemory
        memory = PowerSGDMemory(q_memory=compressor.q_memory, compress_rank=1)
    elif mem == 'residual':
        from grace_dl.torch.memory.residual import ResidualMemory
        memory = ResidualMemory()
    elif mem == 'efsignsgd':
        from grace_dl.torch.memory.efsignsgd import EFSignSGDMemory
        memory = EFSignSGDMemory(lr=0.1)
    else:
        raise NotImplementedError(mem)

    if comm == 'allreduce':
        from grace_dl.torch.communicator.allreduce import Allreduce
        return Allreduce(compressor, memory)
    elif comm == 'allgather':
        from grace_dl.torch.communicator.allgather import Allgather
        return Allgather(compressor, memory, world_size)
    elif comm == 'broadcast':
        from grace_dl.torch.communicator.broadcast import Broadcast
        return Broadcast(compressor, memory, world_size)
    else:
        raise NotImplementedError(comm)
