from __future__ import absolute_import

import numpy as np
import pytest

from pyQCD.core import *


def multiply_einsum(a, b):
    """Multiplies Lattice/Array types together using einsum"""
    return np.einsum('...jk,...kl->...jl', a, b)

@pytest.mark.parametrize(
    "Type,args",
    [(ColourMatrix, ()), (ColourVector, ()),
     (LatticeColourMatrix, (LexicoLayout([8, 4, 4, 4]), 4)),
     (LatticeColourVector, (LexicoLayout([8, 4, 4, 4]), 4))])
class TestMatrixType(object):

    def test_constructor(self, Type, args):
        """Test matrix"""
        mat = Type(*args)
        assert isinstance(mat, Type)

    def test_buffer_protocol(self, Type, args):
        """Test buffer protocol implementation"""
        mat = Type(*args)
        np_mat1 = np.asarray(mat)
        np_mat1.dtype = complex
        np_mat2 = np.asarray(mat)
        np_mat2.dtype = complex
        shape = np_mat1.shape
        for index in np.ndindex(shape):
            assert np_mat1[index] == 0j

        index = tuple(s - 1 if i == 0 else 0 for i, s in enumerate(shape))
        np_mat1[index] = 5.0
        assert np_mat1[index] == 5.0 + 0j
        assert np_mat2[index] == 5.0 + 0j

    def test_as_numpy(self, Type, args):
        """Test as_numpy attribute"""
        mat = Type(*args)
        np_mat1 = mat.as_numpy

        shape = np_mat1.shape
        for index in np.ndindex(shape):
            assert np_mat1[index] == 0j

        if (isinstance(mat, LatticeColourVector) or
            isinstance(mat, LatticeColourMatrix)):
            index = (4, 3, 2, 1, 0)
            np_mat1[index] = np.ones(shape[5:])
            np_mat2 = np.asarray(mat)
            np_mat2.dtype = complex
            assert (np_mat2[1252] == np.ones(shape[5:])).all()

    def test_accessors(self, Type, args):
        """Test __getitem and __setitem__"""
        mat = Type(*args)
        num_dims = len(mat.as_numpy.shape)
        index = (0,) * num_dims
        value = mat[index]

        assert value == 0j

        mat[index] = 1.0

        assert mat.as_numpy[index] == 1.0

    def test_attributes(self, Type, args):
        """Test numpy attribute lookup"""
        mat = Type(*args)

        # Run through a few attributes to compare them with numpy values
        assert mat.shape == mat.as_numpy.shape
        assert mat.size == mat.as_numpy.size
        assert mat.nbytes == mat.as_numpy.nbytes

    def test_random(self, Type, args):
        """Test generation of random ColourMatrix for conformance with SU(N)"""

        if not Type is ColourMatrix:
            return

        mat = Type.random()
        mat_view = mat.as_numpy
        num_colours = mat_view.shape[-1]

        for i in range(num_colours):
            for j in range(num_colours):
                expected_value = 1.0 if i == j else 0.0
                assert np.allclose(np.vdot(mat_view[i], mat_view[j]),
                                   expected_value)

        assert np.allclose(np.linalg.det(mat_view), 1.0)
        assert np.allclose(np.dot(mat_view, np.conj(mat_view.T)),
                           np.identity(num_colours))

    def test_change_layout(self, Type, args):
        """Test change of layout for lattice types"""

        if not Type.__name__.startswith("Lattice"):
            return

        lattice = Type(*args)

        new_layout = EvenOddLayout(args[0].shape)

        lattice.change_layout(new_layout)

        with pytest.raises(ValueError):
            arr = lattice.as_numpy
