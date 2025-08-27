from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

colpack_root = "/h/28/lucahiede/Projects/sparse_hessian_ff/ColPack"


ext = Pybind11Extension(
    "colpackpy",
    ["colpackpy.cpp", "utilities.cpp", "colpackpy2.cpp"],
    include_dirs=[
        f"{colpack_root}/build/cmake/mywork/include/ColPack_headers",
        f"{colpack_root}/src/Utilities",
        f"{colpack_root}/src/GeneralGraphColoring",
        f"{colpack_root}/src/BipartiteGraphBicoloring",
        f"{colpack_root}/src/BipartiteGraphPartialColoring",
        f"{colpack_root}/src/Recovery",
    ],
    library_dirs=[f"{colpack_root}/build/cmake/mywork/lib/shared_library"],
    libraries=["ColPack", "dl"],
    extra_compile_args=["-O3", "-std=c++17"],
)

ext2 = Pybind11Extension(
    "colpackpy2",
    ["colpackpy2.cpp", "utilities.cpp"],
    include_dirs=[
        f"{colpack_root}/build/cmake/mywork/include/ColPack_headers",
        f"{colpack_root}/src/Utilities",
        f"{colpack_root}/src/GeneralGraphColoring",
        f"{colpack_root}/src/BipartiteGraphBicoloring",
        f"{colpack_root}/src/BipartiteGraphPartialColoring",
        f"{colpack_root}/src/Recovery",
    ],
    library_dirs=[f"{colpack_root}/build/cmake/mywork/lib/shared_library"],
    libraries=["ColPack", "dl"],
    extra_compile_args=["-O3", "-std=c++17"],
)

setup(
    name="colpackpy",
    version="0.0.1",
    ext_modules=[ext, ext2],
    cmdclass={"build_ext": build_ext},
)


# Run with: 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/28/lucahiede/Projects/sparse_hessian_ff/ColPack/build/cmake/mywork/lib/shared_library
# python setup.py build_ext --inplace