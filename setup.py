from setuptools import setup, Extension
import os

# Build the extension with the package namespace `cosinefusion.core_backend`.
# The compiled binary will be placed in the package namespace so imports like
# `from cosinefusion import core_backend` resolve directly to the extension.
module = Extension(
    'cosinefusion.core_backend',
    sources=['src/cpp/core_init.cpp'],
    extra_compile_args=['-O3', '-std=c++17']
)

setup(
    ext_modules=[module]
)
