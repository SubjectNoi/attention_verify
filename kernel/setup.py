from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

extra_compile_args = {
    "cxx": [
        "-std=c++17",
        "-O2"
    ],
    "nvcc": [
        "-O2",
        "-std=c++17",
        "-arch=compute_80",
        "-code=sm_80",
        "-lineinfo",
    ],
}

setup(
    name="MyAttn",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="MyAttn",
            sources=[
                "pybind.cpp",
                "my_attention/my_attention.cu"
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
