import os

from distutils.sysconfig import customize_compiler
from distutils.util import get_platform
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):

    def finalize_options(self):
        build_ext.finalize_options(self)

        if not self.distribution.has_c_libraries():
            return

        # Go through individual extensions and set their rpath variable and
        # library search paths

        build_clib = self.get_finalized_command("build_clib")
        libnames = [libname for libname, _ in build_clib.libraries]

        rpaths_abs = [os.path.dirname(build_clib.get_fullpath(lib))
                      for lib in build_clib.libraries]
        rpaths_abs = list(set(rpaths_abs))

        libdirs = [os.path.join(build_clib.build_clib, p)
                   for p in rpaths_abs]

        for ext in self.extensions:
            ext_path = self.get_ext_filename(ext.name)
            rpaths_rel = [os.path.relpath(p, os.path.dirname(ext_path))
                          for p in rpaths_abs]
            ext.runtime_library_dirs.extend(
                ["$ORIGIN/{}".format(p) for p in rpaths_rel])

            ext.libraries.extend(libnames)
            ext.library_dirs.extend(libdirs)

    def run(self):

        # Code below is lovingly spliced together from setuptools and distutils
        # build_ext implementations, minus some code

        # From setuptools:
        old_inplace, self.inplace = self.inplace, 0

        # From distutils...
        from distutils.ccompiler import new_compiler

        # 'self.extensions', as supplied by setup.py, is a list of
        # Extension instances.  See the documentation for Extension (in
        # distutils.extension) for details.
        #
        # For backwards compatibility with Distutils 0.8.2 and earlier, we
        # also allow the 'extensions' list to be a list of tuples:
        #    (ext_name, build_info)
        # where build_info is a dictionary containing everything that
        # Extension instances do except the name, with a few things being
        # differently named.  We convert these 2-tuples to Extension
        # instances as needed.

        if not self.extensions:
            return

        # N.B. Old library directory-setting code from distutils was here...

        # If we were asked to build any C/C++ libraries, make sure that the
        # directory where we put them is in the library search path for
        # linking extensions.

        # Setup the CCompiler object that we'll use to do all the
        # compiling and linking
        self.compiler = new_compiler(compiler=self.compiler,
                                     verbose=self.verbose,
                                     dry_run=self.dry_run,
                                     force=self.force)
        customize_compiler(self.compiler)
        # If we are cross-compiling, init the compiler now (if we are not
        # cross-compiling, init would not hurt, but people may rely on
        # late initialization of compiler even if they shouldn't...)
        if os.name == 'nt' and self.plat_name != get_platform():
            self.compiler.initialize(self.plat_name)

        # And make sure that any compile/link-related options (which might
        # come from the command-line or from the setup script) are set in
        # that CCompiler object -- that way, they automatically apply to
        # all compiling and linking done here.
        if self.include_dirs is not None:
            self.compiler.set_include_dirs(self.include_dirs)
        if self.define is not None:
            # 'define' option is a list of (name,value) tuples
            for (name, value) in self.define:
                self.compiler.define_macro(name, value)
        if self.undef is not None:
            for macro in self.undef:
                self.compiler.undefine_macro(macro)
        if self.libraries is not None:
            self.compiler.set_libraries(self.libraries)
        if self.library_dirs is not None:
            self.compiler.set_library_dirs(self.library_dirs)
        if self.rpath is not None:
            self.compiler.set_runtime_library_dirs(self.rpath)
        if self.link_objects is not None:
            self.compiler.set_link_objects(self.link_objects)

        # Now actually compile and link everything.
        self.build_extensions()

        # From setuptools:
        self.inplace = old_inplace
        if old_inplace:
            self.copy_extensions_to_source()
