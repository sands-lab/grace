def grace_from_params(params):
    import horovod.tensorflow as hvd
    world_size = hvd.size()
    comp = params.get('compressor', 'none')
    mem = params.get('memory', 'none')
    comm = params.get('communicator', 'allreduce')
    if comp == 'adaq':
        from grace_dl.tensorflow.compressor.adaq import AdaqCompressor
        compressor = AdaqCompressor(compress_ratio=0.01)
    elif comp == 'dgc':
        from grace_dl.tensorflow.compressor.dgc import DgcCompressor
        compressor = DgcCompressor(compress_ratio=0.01)
    elif comp == 'efsignsgd':
        from grace_dl.tensorflow.compressor.efsignsgd import EFSignSGDCompressor
        compressor = EFSignSGDCompressor(lr=0.1)
    elif comp == 'fp16':
        from grace_dl.tensorflow.compressor.fp16 import FP16Compressor
        compressor = FP16Compressor()
    elif comp == 'inceptionn':
        from grace_dl.tensorflow.compressor.inceptionn import INCEPTIONNCompressor
        compressor = INCEPTIONNCompressor(error_bound=2e-10)
    elif comp == 'natural':
        from grace_dl.tensorflow.compressor.natural import NaturalCompressor
        compressor = NaturalCompressor()
    elif comp == 'none':
        from grace_dl.tensorflow.compressor.none import NoneCompressor
        compressor = NoneCompressor()
    elif comp == 'onebit':
        from grace_dl.tensorflow.compressor.onebit import OneBitCompressor
        compressor = OneBitCompressor()
    elif comp == 'powersgd':
        from grace_dl.tensorflow.compressor.powersgd import PowerSGDCompressor
        compressor = PowerSGDCompressor(momentum_factor=0.9, world_size=world_size)
    elif comp == 'qsgd':
        from grace_dl.tensorflow.compressor.qsgd import QSGDCompressor
        compressor = QSGDCompressor(quantum_num=64)
    elif comp == 'randomk':
        from grace_dl.tensorflow.compressor.randomk import RandomKCompressor
        compressor = RandomKCompressor(compress_ratio=0.01)
    elif comp == 'signsgd':
        from grace_dl.tensorflow.compressor.signsgd import SignSGDCompressor
        compressor = SignSGDCompressor()
    elif comp == 'signum':
        from grace_dl.tensorflow.compressor.signum import SignumCompressor
        compressor = SignumCompressor(momentum=0.9)
    elif comp == 'sketch':
        from grace_dl.tensorflow.compressor.sketch import SketchCompressor
        compressor = SketchCompressor(quantiles=64)
    elif comp == 'terngrad':
        from grace_dl.tensorflow.compressor.terngrad import TernGradCompressor
        compressor = TernGradCompressor()
    elif comp == 'threshold':
        from grace_dl.tensorflow.compressor.threshold import ThresholdCompressor
        compressor = ThresholdCompressor(threshold=0.01)
    elif comp == 'topk':
        from grace_dl.tensorflow.compressor.topk import TopKCompressor
        compressor = TopKCompressor(compress_ratio=0.01)
    elif comp == 'u8bit':
        from grace_dl.tensorflow.compressor.u8bit import U8bitCompressor
        compressor = U8bitCompressor()
    else:
        raise NotImplementedError(comp)

    if mem == 'dgc':
        from grace_dl.tensorflow.memory.dgc import DgcMemory
        memory = DgcMemory(momentum=0.9, gradient_clipping=False, world_size=world_size)
    elif mem == 'none':
        from grace_dl.tensorflow.memory.none import NoneMemory
        memory = NoneMemory()
    elif mem == 'powersgd':
        from grace_dl.tensorflow.memory.powersgd import PowerSGDMemory
        memory = PowerSGDMemory(q_memory=compressor.q_memory, compress_rank=1,)
    elif mem == 'residual':
        from grace_dl.tensorflow.memory.residual import ResidualMemory
        memory = ResidualMemory()
    elif mem == 'efsignsgd':
        from grace_dl.tensorflow.memory.efsignsgd import EFSignSGDMemory
        memory = EFSignSGDMemory(lr=0.1)
    else:
        raise NotImplementedError(mem)

    if comm == 'allreduce':
        from grace_dl.tensorflow.communicator.allreduce import Allreduce
        return Allreduce(compressor, memory, world_size)
    elif comm == 'allgather':
        from grace_dl.tensorflow.communicator.allgather import Allgather
        return Allgather(compressor, memory, world_size)
    elif comm == 'broadcast':
        from grace_dl.tensorflow.communicator.broadcast import Broadcast
        return Broadcast(compressor, memory, world_size)
    else:
        raise NotImplementedError(comm)
