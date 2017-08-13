from functools import partial
import os
import sys
import warnings

from setuptools import Extension
from setuptools.command.test import test as TestCommand

from pyQCD.utils.build.build_shared_clib import make_library


# Include/library search hints
file_search_paths = ["/usr/include", "/usr/lib",
                     "/usr/local/include", "/usr/local/lib",
                     "/usr/local", "/usr",
                     "/opt", "/"]

# N.B. Only use these lists for flags that are globally compatible (i.e. every
# supported compiler, every supported platform).
compiler_args = ["-std=c++11", "-O3"]
linker_args = []
header_search_files = ["signature_of_eigen3_matrix_library"]

extension_sources = {
    "pyQCD.core.core": ["pyQCD/core/core.cpp"],
    "pyQCD.fermions.fermions": ["pyQCD/fermions/fermions.cpp"],
    "pyQCD.gauge.gauge": ["pyQCD/gauge/gauge.cpp"],
    "pyQCD.algorithms.algorithms": ["pyQCD/algorithms/algorithms.cpp"]
}

library_sources = {
    "pyQCDcore": ["pyQCD/core/layout.cpp"],
    "pyQCDutils": ["pyQCD/utils/matrices.cpp", "pyQCD/utils/random.cpp"]
}


def generate_flags(compiler):
    """Generate flags for given compiler"""

    compiler_flags = ["-std=c++11", "-O3"]
    linker_flags = []
      
    args = compiler.compiler

    if "clang" in args[0] or "clang++" in args[0]:
        compiler_flags.append("-fopenmp=libomp")
        linker_flags.append("-fopenmp=libomp")
    else:
        compiler_flags.append("-fopenmp")
        linker_flags.append("-lgomp")

    return compiler_flags, linker_flags
    
    import IPython; IPython.embed(); raise KeyboardInterrupt


def generate_include_dirs():
    """Locate required include directories"""
    include_dirs = ["./pyQCD"]
    include_dirs.extend(generate_include_paths(file_search_paths,
                                               header_search_files))
    return include_dirs


def generate_libraries(argv):
    """Generate library specifications"""
    include_dirs = generate_include_dirs()
    make_lib = partial(make_library, language="c++", output_dir="lib",
                       undef_macros=["NDEBUG"], include_dirs=include_dirs,
                       extra_compile_args=compiler_args,
                       extra_link_args=linker_args)

    return [make_lib(name, src) for name, src in library_sources.items()]


def generate_extensions(argv):
    """Generate Extension instances to feed to build_ext -fi"""

    include_dirs = generate_include_dirs()
    make_extension = partial(Extension, language="c++", undef_macros=["NDEBUG"],
                             include_dirs=include_dirs,
                             extra_compile_args=compiler_args,
                             extra_link_args=linker_args)

    extensions = [make_extension(module, sources)
                  for module, sources in extension_sources.items()]
    return extensions


def find_file_in_directory(init_dir, filename):
    """Search all directories under init_dir for the specified filename

    Args:
      init_dir (str): The directory path in which to search.
      filename (str): The file name to look for.

    Returns:
      str or NoneType: The directory path in which the file was found.
        None if not found.
    """

    for directory, dirs, filenames in os.walk(init_dir):
        if filename in filenames:
            return directory


def generate_include_paths(search_paths, includes_to_find):
    """Search for the specified files in the specified search paths.

    Args:
      search_paths (list): List of paths to search in.
      includes_to_find (list): List of file names to locate.

    Returns:
      list: List of include directories.
    """

    search_paths = [path for path in search_paths if os.path.exists(path)]

    include_paths = set()

    for filename in includes_to_find:
        found_file = False
        for path in search_paths:
            include_path = find_file_in_directory(path, filename)

            if include_path:
                include_paths.add(include_path)
                found_file = True
                break

        if not found_file:
            warnings.warn("Unable to find include search file: {}"
                          .format(filename))

    return list(include_paths)


# Test command for interfacing with py.test
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments for py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

