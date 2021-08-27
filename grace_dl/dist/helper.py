def grace_from_params(params):
    comp = params.get('compressor', 'none')
    mem = params.get('memory', 'none')
    comm = params.get('communicator', 'allreduce')
    if comp == 'dgc':
        from grace_dl.dist.compressor.dgc import DgcCompressor
        compress_ratio = params.get('compress_ratio', 0.3)
        compressor = DgcCompressor(compress_ratio)
    elif comp == 'efsignsgd':
        from grace_dl.dist.compressor.efsignsgd import EFSignSGDCompressor
        lr = params.get('lr', 0.1)
        compressor = EFSignSGDCompressor(lr)
    elif comp == 'fp16':
        from grace_dl.dist.compressor.fp16 import FP16Compressor
        compressor = FP16Compressor()
    elif comp == 'natural':
        from grace_dl.dist.compressor.natural import NaturalCompressor
        compressor = NaturalCompressor()
    elif comp == 'natural_cuda':
        from grace_dl.dist.compressor.natural import NaturalCompressor_CUDA
        compressor = NaturalCompressor_CUDA()
    elif comp == 'none':
        from grace_dl.dist.compressor.none import NoneCompressor
        compressor = NoneCompressor()
    elif comp == 'onebit':
        from grace_dl.dist.compressor.onebit import OneBitCompressor
        compressor = OneBitCompressor()
    elif comp == 'powersgd':
        from grace_dl.dist.compressor.powersgd import PowerSGDCompressor
        compressor = PowerSGDCompressor()
    elif comp == 'qsgd':
        from grace_dl.dist.compressor.qsgd import QSGDCompressor
        quantum_num = params.get('quantum_num', 127)
        bucket_size = params.get('bucket_size', 128)
        compressor = QSGDCompressor(quantum_num, bucket_size)
    elif comp == 'qsgd_cuda':
        from grace_dl.dist.compressor.qsgd import QSGDCompressor_CUDA
        quantum_num = params.get('quantum_num', 127)
        bucket_size = params.get('bucket_size', 128)
        compressor = QSGDCompressor_CUDA(quantum_num, bucket_size)
    elif comp == 'randomk':
        from grace_dl.dist.compressor.randomk import RandomKCompressor
        compress_ratio = params.get('compress_ratio', 0.3)
        compressor = RandomKCompressor(compress_ratio)
    elif comp == 'signsgd':
        from grace_dl.dist.compressor.signsgd import SignSGDCompressor
        compressor = SignSGDCompressor()
    elif comp == 'signum':
        from grace_dl.dist.compressor.signum import SignumCompressor
        momentum = params.get('momentum', 0.9)
        compressor = SignumCompressor(momentum)
    elif comp == 'terngrad':
        from grace_dl.dist.compressor.terngrad import TernGradCompressor
        compressor = TernGradCompressor()
    elif comp == 'threshold':
        from grace_dl.dist.compressor.threshold import ThresholdCompressor
        threshold = params.get('threshold', 0.01)
        compressor = ThresholdCompressor(threshold)
    elif comp == 'topk':
        from grace_dl.dist.compressor.topk import TopKCompressor
        compress_ratio = params.get('compress_ratio', 0.3)
        kernel = params.get('kernel', 'torch')
        compressor = TopKCompressor(compress_ratio, kernel)
    else:
        raise NotImplementedError(comp)

    if mem == 'dgc':
        from grace_dl.dist.memory.dgc import DgcMemory
        momentum = params.get('momentum', 0.9)
        gradient_clipping = params.get('gradient_clipping', False)
        memory = DgcMemory(momentum, gradient_clipping, params['world_size'])
    elif mem == 'none':
        from grace_dl.dist.memory.none import NoneMemory
        memory = NoneMemory()
    elif mem == 'powersgd':
        from grace_dl.dist.memory.powersgd import PowerSGDMemory
        compress_rank = params.get('compress_rank', 1)
        memory = PowerSGDMemory(compressor.q_memory, compress_rank)
    elif mem == 'residual':
        from grace_dl.dist.memory.residual import ResidualMemory
        memory = ResidualMemory()
    elif mem == 'efsignsgd':
        from grace_dl.dist.memory.efsignsgd import EFSignSGDMemory
        lr = params.get('lr', 0.1)
        memory = EFSignSGDMemory(lr)
    else:
        raise NotImplementedError(mem)

    if comm == 'allreduce':
        from grace_dl.dist.communicator.allreduce import Allreduce
        return Allreduce(compressor, memory, params['world_size'])
    elif comm == 'allgather':
        from grace_dl.dist.communicator.allgather import Allgather
        return Allgather(compressor, memory, params['world_size'])
    elif comm == 'broadcast':
        from grace_dl.dist.communicator.broadcast import Broadcast
        return Broadcast(compressor, memory, params['world_size'])
    elif comm == 'alltoall':
        from grace_dl.dist.communicator.all_to_all import AllToAll
        return AllToAll(compressor, memory, params['world_size'])
    else:
        raise NotImplementedError(comm)
