from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='qsgd_cuda',
    ext_modules=[
        CUDAExtension('qsgd_cuda', [
            'qsgd.cpp',
            'qsgd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
