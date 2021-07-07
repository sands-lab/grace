from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cnat_cuda',
    ext_modules=[
        CUDAExtension('cnat_cuda', [
            'cnat.cpp',
            'cnat_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
