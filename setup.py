from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import glob

ext_modules = [
    Pybind11Extension(
        "liquidity_metrics",
        sorted(glob.glob("src/python/bindings.cpp")),
        include_dirs=["src/cpp"],
        libraries=["liquidity_core", "cuda_metrics"],
        library_dirs=["build"],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="liquidity-metrics",
    version="1.0.0",
    author="Liquidity Metrics Team",
    description="High-performance liquidity metrics calculation using SIMD and CUDA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
    extras_require={
        "dev": ["pytest", "black", "mypy", "flake8"],
    },
    zip_safe=False,
)
