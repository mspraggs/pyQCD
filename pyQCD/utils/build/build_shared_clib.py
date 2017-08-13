"""
I apologise in advance for the below, which is basically the source for the
build_ext and build_clib commands, lovingly hacked about a bit.
"""

import os

from distutils import log
from distutils.core import Command
from distutils.errors import *
from distutils.dir_util import mkpath
from distutils.file_util import copy_file
from distutils.sysconfig import customize_compiler


def make_library(name, source_files, language, output_dir, undef_macros=None,
                 include_dirs=None, extra_compile_args=None,
                 extra_link_args=None):

    return [name, {"sources": source_files, "language": language,
                   "output_dir": output_dir, "undef_macros": undef_macros,
                   "include_dirs": include_dirs,
                   "extra_compile_args": extra_compile_args,
                   "extra_link_args": extra_link_args}]


def show_compilers():
    from distutils.ccompiler import show_compilers
    show_compilers()


class BuildSharedLib(Command):

    description = "build C/C++ libraries used by Python extensions"

    user_options = [
        ('build-clib', 'b',
         "directory to build C/C++ libraries to"),
        ('build-temp', 't',
         "directory to put temporary build by-products"),
        ('debug', 'g',
         "compile with debugging information"),
        ('force', 'f',
         "forcibly build everything (ignore file timestamps)"),
        ('compiler=', 'c',
         "specify the compiler type"),
        ('inplace', 'i',
         "ignore build-lib and put compiled extensions into the source " +
         "directory alongside your pure Python modules"),
        ]

    boolean_options = ['debug', 'force', 'inplace']

    help_options = [
        ('help-compiler', None,
         "list available compilers", show_compilers),
        ]

    def initialize_options(self):
        self.build_clib = None
        self.build_temp = None
        self.build_lib = None

        self.package = None

        # List of libraries to build
        self.libraries = None

        # Compilation options for all libraries
        self.include_dirs = None
        self.define = None
        self.undef = None
        self.debug = None
        self.force = 0
        self.inplace = False
        self.compiler = None

    def finalize_options(self):
        # Set undefined options. Note that shared libraries are compiled into
        # path specified by build_lib, not build_temp
        self.set_undefined_options('build',
                                   ("build_lib", "build_lib"),
                                   ('build_lib', 'build_clib'),
                                   ('build_temp', 'build_temp'),
                                   ('compiler', 'compiler'),
                                   ('debug', 'debug'),
                                   ('force', 'force'))

        if self.package is None:
            self.package = self.distribution.ext_package
        
        self.libraries = self.distribution.libraries
        if self.libraries:
            self.check_library_list(self.libraries)

        if self.include_dirs is None:
            self.include_dirs = self.distribution.include_dirs or []
        if isinstance(self.include_dirs, str):
            self.include_dirs = self.include_dirs.split(os.pathsep)

    def run(self):

        from pyQCD.utils.build import generate_flags
        
        if not self.libraries:
            return

        # Yech -- this is cut 'n pasted from build_ext.py!
        from distutils.ccompiler import new_compiler
        self.compiler = new_compiler(compiler=self.compiler,
                                     dry_run=self.dry_run,
                                     force=self.force)
        customize_compiler(self.compiler)

        if self.include_dirs is not None:
            self.compiler.set_include_dirs(self.include_dirs)
        if self.define is not None:
            # 'define' option is a list of (name,value) tuples
            for (name,value) in self.define:
                self.compiler.define_macro(name, value)
        if self.undef is not None:
            for macro in self.undef:
                self.compiler.undefine_macro(macro)

        compile_args, link_args = generate_flags(self.compiler)

        for lib_name, build_info in self.libraries:
            build_info["extra_compile_args"].extend(compile_args)
            build_info["extra_link_args"].extend(link_args)

        old_inplace, self.inplace = self.inplace, 0
        self.build_libraries(self.libraries)
        self.inplace = old_inplace
        if old_inplace:
            self.copy_libraries_to_output()


    def check_library_list(self, libraries):
        """Ensure that the list of libraries is valid.

        `library` is presumably provided as a command option 'libraries'.
        This method checks that it is a list of 2-tuples, where the tuples
        are (library_name, build_info_dict).

        Raise DistutilsSetupError if the structure is invalid anywhere;
        just returns otherwise.
        """
        if not isinstance(libraries, list):
            raise DistutilsSetupError(
                "'libraries' option must be a list of tuples")

        for lib in libraries:
            if not isinstance(lib, tuple) and len(lib) != 2:
                raise DistutilsSetupError(
                    "each element of 'libraries' must a 2-tuple")

            name, build_info = lib

            if not isinstance(name, str):
                raise DistutilsSetupError(
                    "first element of each tuple in 'libraries' must be a "
                    "string (the library name)")
            if '/' in name or (os.sep != '/' and os.sep in name):
                raise DistutilsSetupError(
                    "bad library name {}: may not contain directory separators"
                    .format(lib[0]))

            if not isinstance(build_info, dict):
                raise DistutilsSetupError(
                    "second element of each tuple in 'libraries' "
                    "must be a dictionary (build info)")

    def get_library_names(self):
        # Assume the library list is valid -- 'check_library_list()' is
        # called from 'finalize_options()', so it should be!
        if not self.libraries:
            return None

        lib_names = []
        for (lib_name, build_info) in self.libraries:
            lib_names.append(lib_name)
        return lib_names

    def get_source_files(self):
        self.check_library_list(self.libraries)
        filenames = []
        for (lib_name, build_info) in self.libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                    "in 'libraries' option (library '{}'), "
                    "'sources' must be present and must be "
                    "a list of source filenames".format(lib_name))

            filenames.extend(sources)
        return filenames

    def build_libraries(self, libraries):

        for (lib_name, build_info) in libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                    "in 'libraries' option (library '{}'), "
                    "'sources' must be present and must be "
                    "a list of source filenames".format(lib_name))
            sources = list(sources)
            extra_args = build_info.get("extra_compile_args") or []
            filename = self.compiler.library_filename(lib_name, "shared")
            
            log.info("building '{}' library".format(lib_name))

            # First, compile the source code to object files in the library
            # directory.  (This should probably change to putting object
            # files in a temporary build directory.)
            macros = build_info.get('macros')
            include_dirs = build_info.get('include_dirs')
            objects = self.compiler.compile(sources,
                                            output_dir=self.build_temp,
                                            macros=macros,
                                            include_dirs=include_dirs,
                                            extra_postargs=extra_args,
                                            debug=self.debug)

            # Link objects into a shared object library

            output_dir = os.path.join(
                self.build_clib, os.path.dirname(
                    self.get_fullpath((lib_name, build_info))))

            mkpath(output_dir)

            self.compiler.link_shared_object(objects, filename,
                                             output_dir=output_dir,
                                             debug=self.debug)

    def get_fullpath(self, lib):

        from distutils.ccompiler import new_compiler
        compiler = new_compiler()

        libname, info = lib
        name = self.distribution.get_name()
        filename = compiler.library_filename(libname, "shared")

        return os.path.join(name, info["output_dir"], filename)

    def copy_libraries_to_output(self):

        for lib_name, build_info in self.libraries:
            dest_filename = self.get_fullpath((lib_name, build_info))
            src_filename = os.path.join(self.build_clib, dest_filename)

            mkpath(os.path.dirname(dest_filename))

            # Always copy, even if source is older than destination, to ensure
            # that the right extensions for the current Python/platform are
            # used.
            copy_file(
                src_filename, dest_filename, verbose=self.verbose,
                dry_run=self.dry_run
            )
