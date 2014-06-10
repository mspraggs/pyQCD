from setuptools import setup, Extension, find_packages
from setuptools.command import build_ext
import os
import io

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.md')


class my_build_ext(build_ext.build_ext):
    def build_extension(self, ext):
        import shutil
        import os.path
        try:
            import pyQCD.core.kernel.lattice
            module_file = pyQCD.core.kernel.lattice.__file__
            shutil.copyfile(module_file,
                            self.get_ext_fullpath(ext.name))
        except ImportError:
            print("Notice: lattice.so binary missing. Installing analysis modules only.")

# Exclude the lattice.py and simulation.py files if lattice.so
# hasn't been built.

setup(
    name='pyQCD',
    version='',
    packages=find_packages(exclude=["*test*"]),
    url='http://github.com/mspraggs/pyqcd/',
    license='Apache Software License',
    author='Matt Spraggs',
    install_requires=['numpy', 'scipy'],
    author_email='matthew.spraggs@gmail.com',
    description='pyQCD provides a Python library for running coarse lattice '
    'QCD simulations on desktop and workstation computers.',
    long_description=long_description,
    cmdclass = {'build_ext': my_build_ext},
    ext_modules = [Extension('pyQCD/core/kernel/lattice',
                             sources = ['pyQCD/core/kernel/src/wrapper.cpp'])],
    package_dir={'': '.'},
    package_data={
        'pyQCD.codegen': ['templates/*.cpp', 'templates/*.hpp'],
        },
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: C++',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        ],
)
