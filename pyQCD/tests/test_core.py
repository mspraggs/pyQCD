from __future__ import absolute_import

import numpy as np
import pytest

from pyQCD.core import *


def multiply_einsum(a, b):
    """Multiplies Lattice/Array types together using einsum"""
    return np.einsum('...jk,...kl->...jl', a, b)

@pytest.mark.parametrize(
    "Type,multiply,args",
    [(ColourMatrix, np.dot, ()), (ColourVector, None, ()),
     (LatticeColourMatrix, multiply_einsum, ([8, 4, 4, 4], 4)),
     (LatticeColourVector, None, ([8, 4, 4, 4], 4))])
class TestMatrixType(object):

    def test_constructor(self, Type, multiply, args):
        """Test matrix"""
        mat = Type(*args)
        assert isinstance(mat, Type)

    def test_buffer_protocol(self, Type, multiply, args):
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

    def test_as_numpy(self, Type, multiply, args):
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

    def test_mul(self, Type, multiply, args):
        """Test multiplications"""
        shape = Type(*args).as_numpy.shape
        mat1_data = np.random.rand(*shape)
        mat2_data = np.random.rand(*shape)
        mat1, mat2 = Type(*args), Type(*args)
        mat1.as_numpy = mat1_data
        mat2.as_numpy = mat2_data

        if multiply:
            mat3_data = multiply(mat1_data, mat2_data)
            mat3 = mat1 * mat2
            assert np.allclose(mat3.as_numpy, mat3_data)

    @pytest.mark.parametrize("op", [lambda x, y: x + y, lambda x, y: x - y])
    def test_add(self, Type, multiply, args, op):
        """Test addition"""
        shape = Type(*args).as_numpy.shape
        mat1_data = np.random.rand(*shape)
        mat2_data = np.random.rand(*shape)
        mat3_data = op(mat1_data, mat2_data)
        mat1, mat2 = Type(*args), Type(*args)
        mat1.as_numpy = mat1_data
        mat2.as_numpy = mat2_data
        mat3 = op(mat1, mat2)
        assert np.allclose(mat3.as_numpy, mat3_data)
