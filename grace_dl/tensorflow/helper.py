def grace_from_params(params):
    import horovod.tensorflow as hvd
    world_size = hvd.size()
    comp = params.get('compressor', 'none')
    mem = params.get('memory', 'none')
    comm = params.get('communicator', 'allreduce')
    if comp == 'adaq':
        from grace_dl.tensorflow.compressor.adaq import AdaqCompressor
        compress_ratio = params.get('compress_ratio', 0.3)
        compressor = AdaqCompressor(compress_ratio)
    elif comp == 'dgc':
        from grace_dl.tensorflow.compressor.dgc import DgcCompressor
        compress_ratio = params.get('compress_ratio', 0.3)
        compressor = DgcCompressor(compress_ratio)
    elif comp == 'efsignsgd':
        from grace_dl.tensorflow.compressor.efsignsgd import EFSignSGDCompressor
        lr = params.get('lr', 0.1)
        compressor = EFSignSGDCompressor(lr)
    elif comp == 'fp16':
        from grace_dl.tensorflow.compressor.fp16 import FP16Compressor
        compressor = FP16Compressor()
    elif comp == 'inceptionn':
        from grace_dl.tensorflow.compressor.inceptionn import INCEPTIONNCompressor
        error_bound = params.get('error_bound', 2e-10)
        compressor = INCEPTIONNCompressor(error_bound)
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
        momentum_factor = params.get('momentum_factor', 0.9)
        compressor = PowerSGDCompressor(momentum_factor, world_size)
    elif comp == 'qsgd':
        from grace_dl.tensorflow.compressor.qsgd import QSGDCompressor
        quantum_num = params.get('quantum_num', 127)
        compressor = QSGDCompressor(quantum_num)
    elif comp == 'randomk':
        from grace_dl.tensorflow.compressor.randomk import RandomKCompressor
        compress_ratio = params.get('compress_ratio', 0.3)
        compressor = RandomKCompressor(compress_ratio)
    elif comp == 'signsgd':
        from grace_dl.tensorflow.compressor.signsgd import SignSGDCompressor
        compressor = SignSGDCompressor()
    elif comp == 'signum':
        from grace_dl.tensorflow.compressor.signum import SignumCompressor
        momentum = params.get('momentum', 0.9)
        compressor = SignumCompressor(momentum)
    elif comp == 'sketch':
        from grace_dl.tensorflow.compressor.sketch import SketchCompressor
        quantiles = params.get('quantiles', 64)
        compressor = SketchCompressor(quantiles)
    elif comp == 'terngrad':
        from grace_dl.tensorflow.compressor.terngrad import TernGradCompressor
        compressor = TernGradCompressor()
    elif comp == 'threshold':
        from grace_dl.tensorflow.compressor.threshold import ThresholdCompressor
        threshold = params.get('threshold', 0.01)
        compressor = ThresholdCompressor(threshold)
    elif comp == 'topk':
        from grace_dl.tensorflow.compressor.topk import TopKCompressor
        compress_ratio = params.get('compress_ratio', 0.3)
        compressor = TopKCompressor(compress_ratio)
    elif comp == 'u8bit':
        from grace_dl.tensorflow.compressor.u8bit import U8bitCompressor
        compressor = U8bitCompressor()
    else:
        raise NotImplementedError(comp)

    if mem == 'dgc':
        from grace_dl.tensorflow.memory.dgc import DgcMemory
        momentum = params.get('momentum', 0.9)
        gradient_clipping = params.get('gradient_clipping', False)
        memory = DgcMemory(momentum, gradient_clipping, world_size)
    elif mem == 'none':
        from grace_dl.tensorflow.memory.none import NoneMemory
        memory = NoneMemory()
    elif mem == 'powersgd':
        from grace_dl.tensorflow.memory.powersgd import PowerSGDMemory
        compress_rank = params.get('compress_rank', 1)
        memory = PowerSGDMemory(compressor.q_memory, compress_rank)
    elif mem == 'residual':
        from grace_dl.tensorflow.memory.residual import ResidualMemory
        memory = ResidualMemory()
    elif mem == 'efsignsgd':
        from grace_dl.tensorflow.memory.efsignsgd import EFSignSGDMemory
        lr = params.get('lr', 0.1)
        memory = EFSignSGDMemory(lr)
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
