"""Here we define some convenience functions to use within SConstruct files"""

import os
import re
import subprocess
import sys

def gen_common_locations(platform=None):
    """Generate common header/library locations for the current platform"""

    paths = []
    user_path = os.path.expanduser("~")
    if platform == "posix":
        basic_paths = ["/usr",
                      "/usr/local",
                      "/opt",
                      "/opt/local",
                      "/sw",
                      "{}/.local".format(user_path),
                      "{}/soft".format(user_path)]
        paths.extend([os.path.join(p, "include") for p in basic_paths])
        paths.extend([os.path.join(p, "lib") for p in basic_paths])
        paths.extend([os.path.join(p, "lib64") for p in basic_paths])
        if sys.platform == "linux2":
            paths.extend(["/usr/lib/x86_64-linux-gnu",
                          "/usr/lib/i386-linux-gnu"])
    elif platform == "win32":
        paths.append("C:\Windows\system32")
    paths.append(user_path)
    return paths


def recursive_match(search_dir, path_to_find):
    """Look recursively through search_dir, looking for path_to_find.

    Looks at the tip of each path in the tree until a match to path_to_find is
    found, then returns the directory in which this was found.

    Args:
      search_dir (str): Root directory to look in.
      path_to_find (list): Path segment to look for.

    Returns:
      str: The first directory in which path_to_find exists as a relative path.

      Returns None if no path is found.
    """

    path_to_find_elems = path_to_find.split(os.path.sep)
    num_nodes = len(path_to_find_elems)

    for dpath, dnames, fnames in os.walk(search_dir):
        for fname in fnames:
            path_elems = dpath.split(os.path.sep) + [fname]
            if path_elems[-num_nodes:] == path_to_find_elems:
                return os.path.sep.join(path_elems[:-num_nodes])
    return None


def find_file(search_paths, filename):
    """Find the specified file in the given search paths.

    Args:
      search_paths (list): List of directories in which to search.
      filename (str): The file path to find.

    Returns:
      str: The directory in which the path was found (None if not found).
    """

    for search_path in search_paths:
        path = recursive_match(search_path, filename)
        if path:
            return path

def _try_run_eigen(context):
    """Tries to build and run a simple Eigen 3 program."""

    source_code = """#include <Eigen/Eigen>
    int main(int argc, char* argv) {
    }
    """
    return context.TryRun(source_code, ".cpp")


def check_eigen(context):
    """Custom SCons check for Eigen 3.

    Args:
      context (SCons.SConf.CheckContext): SCons context for running checks and
        so forth.

    Returns:
      tuple: Result of context.TryRun.
    """
    context.Display("Checking for Eigen 3... ")

    # Try our best to find eigen include path in the common include
    # locations
    common_paths = gen_common_locations(context.env['PLATFORM'])
    eigen_include_path = find_file(common_paths, "Eigen/Eigen")
    context.env.Append(CPPPATH=[eigen_include_path])
    
    result = _try_run_eigen(context)
    return context.Result(result)


def get_executable_path(executable_name):
    """Find the path of an executable by looking at the PATH variable.

    Args:
      executable_name (str): The filename of the executable to find.

    Returns:
      str: The directory in which the executable resides (None if not found).
    """

    path_sep = ";" if sys.platform == "win32" else ":"
    if sys.platform == "win32" and os.path.splitext(executable_name) != ".exe":
        executable_name += ".exe"

    paths = os.environ['PATH'].split(path_sep)
    for path in paths:
        exec_path = os.path.join(path, executable_name)
        if os.path.exists(exec_path):
            return exec_path


def _try_mpi_run(context):
    """Tries to build and run a simple MPI program in the given scons context.
    """

    mpi_code = """#include <mpi.h>
    int main(int argc, char* argv[])
    {
      MPI_Init(&argc, &argv);
      MPI_Finalize();
    }
    """

    return context.TryRun(mpi_code, ".cpp")


def _interrogate_compiler(compiler, arg):
    """Return output from the compiler, or None if the call failed"""
    try:
        subprocess.check_call([compiler, arg], stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return
    else:
        # Legacy support for extracting output
        p = subprocess.Popen([compiler, arg],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        return " ".join([o.rstrip() for o in p.communicate()]) + " "


def mpi_check(context):
    """Custom SCons check for MPI.

    Args:
      context (SCons.SConf.CheckContext): SCons context for running checks and
        so forth.

    Returns:
      tuple: Result of context.TryRun.
    """

    context.Display("Checking for MPI... ")
    # Backup the Environment
    env_backup = context.env.Clone()

    # The following is based off my interpretation of FindMPI.cmake
    # TODO: Support M$- *ahem*, I mean MS-MPI
    # First we need to find a compiler
    import mpi4py
    try:
        mpi_compiler = mpi4py.get_config()['mpicxx']
    except KeyError:
        # Fall back to the native compiler in the worse case (last element).
        trial_compilers = ["mpicxx", "mpiCC", "mpcxx", "mpCC", "mpic++",
                           "mpc++", "mpicxx_r", "mpiCC_r", "mpcxx_r", "mpCC_r",
                           "mpic++_r", "mpc++_r", context.env['CXX']]

        for comp in trial_compilers:
            mpi_compiler = get_executable_path(comp)
            if mpi_compiler:
                break
        try:
            context.Display("Using MPI compiler: {}".format(mpi_compiler))
        except NameError:
            context.Display("No MPI compiler found.")
            return

    # Now figure out what flags we need by trying some compiler commands.
    # First try OpenMPI compiler arguments.
    mpi_cmdline = _interrogate_compiler(mpi_compiler, "-showme:compile")
    if mpi_cmdline:
        for arg in ["-showme:link", "-showme:incdirs", "-showme:libdirs"]:
            try:
                mpi_cmdline += _interrogate_compiler(mpi_compiler, arg)
            except TypeError:
                mpi_cmdline = None
                break
    # Next try older versions of LAM-MPI
    if not mpi_cmdline:
        mpi_cmdline = _interrogate_compiler(mpi_compiler, "-showme")
    # Now try MVAPICH
    if not mpi_cmdline:
        mpi_cmdline = _interrogate_compiler(mpi_compiler, "-compile-info")
        try:
            mpi_cmdline += _interrogate_compiler(mpi_compiler, "-link-info")
        except TypeError:
            mpi_cmdline = None
    # Now try MPICH
    if not mpi_cmdline:
        mpi_cmdline = _interrogate_compiler(mpi_compiler, "-show")

    compile_flags = ["-DUSE_MPI"]
    link_flags = []
    include_paths = []
    library_paths = []
    libraries = []

    if mpi_cmdline:
        # Assuming we got some output, we now need to parse that into summit
        # useful.
        # TODO: MS compiler support
        mpi_cmdline = mpi_cmdline.rstrip()
        escaped_sep = re.escape(os.path.sep)
        multi_sep_pattern = "{}+".format(escaped_sep)
        mpi_cmdline = re.sub(multi_sep_pattern, os.path.sep, mpi_cmdline)

        # TODO: DRY principle

        incpath_pattern = r"(^| )-I([^\" ]+|\"[^\"]+\")"
        incpath_matches = re.findall(incpath_pattern, mpi_cmdline)
        include_paths.extend([p for _, p in incpath_matches])
        mpi_cmdline = re.sub(incpath_pattern, "", mpi_cmdline)

        libpath_pattern = r"(^| |-Wl,)-L([^\" ]+|\"[^\"]+\")"
        libpath_matches = re.findall(libpath_pattern, mpi_cmdline)
        library_paths.extend([p for _, p in libpath_matches])
        mpi_cmdline = re.sub(libpath_pattern, "", mpi_cmdline)

        library_pattern = r"(^| )-l([^\" ]+|\"[^\"]+\")"
        library_matches = re.findall(library_pattern, mpi_cmdline)
        libraries.extend([l for _, l in library_matches])
        mpi_cmdline = re.sub(library_pattern, "", mpi_cmdline)

        lflag_pattern = r"(^| )(-Wl,[^\" ]+|\"[^\"]+\")"
        lflags_matches = re.findall(lflag_pattern, mpi_cmdline)
        link_flags.extend([f for _, f in lflags_matches])
        mpi_cmdline = re.sub(lflag_pattern, "", mpi_cmdline)

        cflag_pattern = r"(^| )(-[^\" ]+|\"[^\"]+\")"
        cflags_matches = re.findall(cflag_pattern, mpi_cmdline)
        compile_flags.extend([f for _, f in cflags_matches])
        mpi_cmdline = re.sub(cflag_pattern, "", mpi_cmdline)

    context.env.Replace(CXX=mpi_compiler)
    context.env.Append(CXXFLAGS=compile_flags, LINKFLAGS=link_flags,
                       CPPPATH=include_paths, LIBPATH=library_paths,
                       LIBS=libraries)

    result = _try_mpi_run(context)
    context.Result(result)
    return result
