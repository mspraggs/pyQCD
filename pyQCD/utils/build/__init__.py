import os
import sys
import warnings

from setuptools.command.test import test as TestCommand


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

