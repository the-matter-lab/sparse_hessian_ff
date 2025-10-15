import os
import subprocess
import shutil
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools.command.build_ext import build_ext as _build_ext

# Dynamically determine the ColPack root directory
colpack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ColPack'))

# Update the extension module configuration
ext = Pybind11Extension(
    "colpackpy.coloring",
    ["src/colpackpy/cpp/coloring.cpp", "src/colpackpy/cpp/utilities.cpp"],
    include_dirs=[
        os.path.join(colpack_root, 'inc'),
        os.path.join(colpack_root, 'src/Utilities'),
        os.path.join(colpack_root, 'src/GeneralGraphColoring'),
        os.path.join(colpack_root, 'src/BipartiteGraphBicoloring'),
        os.path.join(colpack_root, 'src/BipartiteGraphPartialColoring'),
        os.path.join(colpack_root, 'src/Recovery'),
    ],
    library_dirs=[os.path.join(colpack_root, 'build/cmake/mywork')],
    libraries=["ColPack", "dl"],
    extra_compile_args=["-O3", "-std=c++17"],
    extra_link_args=["-Wl,-rpath='$ORIGIN'"],
)

class CMakeBuild(_build_ext):
    def run(self):
        # Ensure CMake is installed
        try:
            subprocess.check_call(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions))

        # Configure and build ColPack
        colpack_build_dir = os.path.join(colpack_root, 'build', 'cmake', 'mywork')
        if not os.path.exists(colpack_build_dir):
            os.makedirs(colpack_build_dir)
        subprocess.check_call(['cmake', os.path.join(colpack_root, 'build', 'cmake')], cwd=colpack_build_dir)
        subprocess.check_call(['make'], cwd=colpack_build_dir)

        # Copy shared libraries to the build directory
        lib_source = os.path.join(colpack_build_dir, 'libColPack.so')
        lib_source_versioned = os.path.join(colpack_build_dir, 'libColPack.so.0')
        
        # Create the colpackpy package directory in build
        build_lib = self.build_lib
        colpackpy_dir = os.path.join(build_lib, 'colpackpy')
        os.makedirs(colpackpy_dir, exist_ok=True)

        # Copy shared libraries to the package directory
        if os.path.exists(lib_source):
            shutil.copy2(lib_source, colpackpy_dir)
        if os.path.exists(lib_source_versioned):
            shutil.copy2(lib_source_versioned, colpackpy_dir)
        
        # For editable installs, also copy to source directory
        source_dir = os.path.join(os.path.dirname(__file__), 'src', 'colpackpy')
        if source_dir != colpackpy_dir:
            if os.path.exists(lib_source):
                shutil.copy2(lib_source, source_dir)
            if os.path.exists(lib_source_versioned):
                shutil.copy2(lib_source_versioned, source_dir)
            
        # Now run the original build_ext command
        super().run()
        
        # After building, patch the rpath of the extension module
        source_dir = os.path.join(os.path.dirname(__file__), 'src', 'colpackpy')
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            if os.path.exists(ext_path):
                try:
                    # Try to set rpath using patchelf if available
                    subprocess.run(['patchelf', '--set-rpath', '$ORIGIN', ext_path], check=False)
                except FileNotFoundError:
                    # patchelf not available, try chrpath
                    try:
                        subprocess.run(['chrpath', '-r', '$ORIGIN', ext_path], check=False)
                    except FileNotFoundError:
                        pass  # Neither tool available
                
                # For editable installs, copy extension module to source directory
                if self.inplace:
                    ext_filename = os.path.basename(ext_path)
                    source_ext_path = os.path.join(source_dir, ext_filename)
                    if ext_path != source_ext_path:
                        shutil.copy2(ext_path, source_ext_path)
                        # Also patch the rpath of the source copy
                        try:
                            subprocess.run(['patchelf', '--set-rpath', '$ORIGIN', source_ext_path], check=False)
                        except FileNotFoundError:
                            try:
                                subprocess.run(['chrpath', '-r', '$ORIGIN', source_ext_path], check=False)
                            except FileNotFoundError:
                                pass

setup(
    name="colpackpy",
    version="0.0.1",
    packages=["colpackpy"],
    package_dir={"colpackpy": "src/colpackpy"},
    ext_modules=[ext],
    cmdclass={"build_ext": CMakeBuild},
    package_data={
        'colpackpy': ['*.so', '*.so.*', '*.py'],
    },
    include_package_data=True,
    zip_safe=False,
)

