from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GNNAdvisor',
    ext_modules=[
        CUDAExtension(
            name='GNNAdvisor',
            sources=[
                'GNNAdvisor.cpp',
                'GNNAdvisor_kernel.cu'
            ],
            libraries=['cuda'],
            include_dirs=[
                "/usr/local/cuda/include",
                "/usr/local/cuda-11.6/targets/x86_64-linux/include"],
            library_dirs=['/usr/local/cuda-11.6/lib64']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
