def grace_from_params(params):
    import horovod.torch as hvd
    world_size = hvd.size()
    comp = params.get('compressor', 'none')
    mem = params.get('memory', 'none')
    comm = params.get('communicator', 'allreduce')
    if comp == 'dgc':
        from grace_dl.torch.compressor.dgc import DgcCompressor
        compress_ratio = params.get('compress_ratio', 0.3)
        compressor = DgcCompressor(compress_ratio)
    elif comp == 'efsignsgd':
        from grace_dl.torch.compressor.efsignsgd import EFSignSGDCompressor
        lr = params.get('lr', 0.1)
        compressor = EFSignSGDCompressor(lr)
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
        quantum_num = params.get('quantum_num', 127)
        compressor = QSGDCompressor(quantum_num)
    elif comp == 'randomk':
        from grace_dl.torch.compressor.randomk import RandomKCompressor
        compress_ratio = params.get('compress_ratio', 0.3)
        compressor = RandomKCompressor(compress_ratio)
    elif comp == 'signsgd':
        from grace_dl.torch.compressor.signsgd import SignSGDCompressor
        compressor = SignSGDCompressor()
    elif comp == 'signum':
        from grace_dl.torch.compressor.signum import SignumCompressor
        momentum = params.get('momentum', 0.9)
        compressor = SignumCompressor(momentum)
    elif comp == 'terngrad':
        from grace_dl.torch.compressor.terngrad import TernGradCompressor
        compressor = TernGradCompressor()
    elif comp == 'threshold':
        from grace_dl.torch.compressor.threshold import ThresholdCompressor
        threshold = params.get('threshold', 0.01)
        compressor = ThresholdCompressor(threshold)
    elif comp == 'topk':
        from grace_dl.torch.compressor.topk import TopKCompressor
        compress_ratio = params.get('compress_ratio', 0.3)
        compressor = TopKCompressor(compress_ratio)
    else:
        raise NotImplementedError(comp)

    if mem == 'dgc':
        from grace_dl.torch.memory.dgc import DgcMemory
        momentum = params.get('momentum', 0.9)
        gradient_clipping = params.get('gradient_clipping', False)
        memory = DgcMemory(momentum, gradient_clipping)
    elif mem == 'none':
        from grace_dl.torch.memory.none import NoneMemory
        memory = NoneMemory()
    elif mem == 'powersgd':
        from grace_dl.torch.memory.powersgd import PowerSGDMemory
        compress_rank = params.get('compress_rank', 1)
        memory = PowerSGDMemory(compressor.q_memory, compress_rank)
    elif mem == 'residual':
        from grace_dl.torch.memory.residual import ResidualMemory
        memory = ResidualMemory()
    elif mem == 'efsignsgd':
        from grace_dl.torch.memory.efsignsgd import EFSignSGDMemory
        lr = params.get('lr', 0.1)
        memory = EFSignSGDMemory(lr)
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
