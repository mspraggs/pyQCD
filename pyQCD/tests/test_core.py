from __future__ import absolute_import

import numpy as np
import pytest

from pyQCD.core import *


class TestLexicoLayout(object):

    def test_constructor(self):
        """Test constructor"""
        for shape in [[4, 4, 4, 4], (4, 4, 4, 4)]:
            layout = LexicoLayout(shape)
            assert isinstance(layout, LexicoLayout)
            assert layout.num_dims == 4
            assert layout.volume == 4**4
            assert layout.shape == (4, 4, 4, 4)

    def test_get_array_index(self):
        """Test array index lookup"""
        layout = LexicoLayout([4, 4, 4, 4])
        assert layout.get_array_index([0, 0, 0, 0]) == 0
        assert layout.get_array_index((0, 0, 0, 1)) == 1
        assert layout.get_array_index([0, 0, 1, 0]) == 4
        assert layout.get_array_index((0, 1, 0, 0)) == 16
        assert layout.get_array_index([1, 0, 0, 0]) == 64
        assert layout.get_array_index((3, 3, 3, 3)) == 255

        for i in range(layout.volume):
            assert layout.get_array_index(i) == i

    def test_get_site_index(self):
        """Test site index lookup"""
        layout = LexicoLayout((4, 4, 4, 4))
        for i in range(layout.volume):
            assert layout.get_site_index(i) == i


class TestComplex(object):

    def test_constructor(self):
        """Test constructor"""
        z = Complex(1.0, 2.0)
        assert isinstance(z, Complex)

    def test_real(self):
        """Test real part property"""
        z = Complex(1.0, 2.0)
        assert z.real == 1.0

    def test_imag(self):
        """Test real part property"""
        z = Complex(1.0, 2.0)
        assert z.imag == 2.0

    def test_to_complex(self):
        """Test Complex to python complex conversion"""
        z = Complex(1.0, 2.0)
        assert z.to_complex() == 1.0 + 2.0j


def multiply_einsum(a, b):
    """Multiplies Lattice/Array types together using einsum"""
    return np.einsum('...jk,...kl->...jl', a, b)

layout = LexicoLayout((8, 4, 4, 4))
@pytest.mark.parametrize(
    "Type,multiply,args",
    [(ColourMatrix, np.dot, ()), (ColourVector, None, ()),
     (LatticeColourMatrix, multiply_einsum, (layout,)),
     (LatticeColourVector, None, (layout,))])
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

    def test_mul(self, Type, multiply, args):
        """Test multiplications"""
        print(Type, args)
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
