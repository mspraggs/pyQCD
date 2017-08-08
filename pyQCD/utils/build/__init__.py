import sys

from setuptools.command.test import test as TestCommand


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

