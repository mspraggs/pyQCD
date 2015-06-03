from __future__ import absolute_import

import numpy as np
import pytest

from pyQCD.core import *


class TestLexicoLayout(object):

    def test_constructor(self):
        """Test constructor"""
        layout = LexicoLayout([4, 4, 4, 4])
        assert isinstance(layout, LexicoLayout)
        layout = LexicoLayout((4, 4, 4, 4))
        assert isinstance(layout, LexicoLayout)

    def test_get_array_index(self):
        """Test array index lookup"""
        layout = LexicoLayout([4, 4, 4, 4])
        assert layout.get_array_index([0, 0, 0, 0]) == 0
        assert layout.get_array_index((0, 0, 0, 1)) == 1
        assert layout.get_array_index([0, 0, 1, 0]) == 4
        assert layout.get_array_index((0, 1, 0, 0)) == 16
        assert layout.get_array_index([1, 0, 0, 0]) == 64
        assert layout.get_array_index((3, 3, 3, 3)) == 255

        for i in range(layout.volume()):
            assert layout.get_array_index(i) == i

    def test_get_site_index(self):
        """Test site index lookup"""
        layout = LexicoLayout((4, 4, 4, 4))
        for i in range(layout.volume()):
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


@pytest.mark.parametrize("Matrix", [ColourMatrix, ColourVector])
class TestMatrixType(object):

    def test_constructor(self, Matrix):
        """Test matrix"""
        mat = Matrix()
        assert isinstance(mat, Matrix)
        mat_values = np.arange(np.prod(mat.shape)).reshape(mat.shape).tolist()
        mat = Matrix(mat_values)
        mat_values = np.array(mat_values)
        assert isinstance(mat, Matrix)

        for index in np.ndindex(mat.shape):
            assert mat[index] == mat_values[index] + 0j

        mat_values = np.arange(np.prod(mat.shape)).reshape(mat.shape)
        mat = Matrix(mat_values)
        assert isinstance(mat, Matrix)

        for index in np.ndindex(mat.shape):
            assert mat[index] == mat_values[index]

        with pytest.raises(IndexError):
            mat = Matrix(np.zeros(tuple(20 for i in mat.shape)))

    def test_boundscheck(self, Matrix):
        """Test bounds checking for matrix"""
        mat = Matrix()
        with pytest.raises(IndexError):
            x = mat[3, 3]
        with pytest.raises(IndexError):
            mat[3, 3] = 4

    def test_zeros(self, Matrix):
        """Test zeros static function"""
        mat = Matrix.zeros()
        assert isinstance(mat, Matrix)

        for index in np.ndindex(mat.shape):
            assert mat[index] == 0.0j

    def test_ones(self, Matrix):
        """Test ones static function"""
        mat = Matrix.ones()
        assert isinstance(mat, Matrix)

        for index in np.ndindex(mat.shape):
            assert mat[index] == 1.0 + 0.0j

    def test_identity(self, Matrix):
        """Test identity static function"""
        if len(Matrix.shape) == 1 or Matrix.shape[0] != Matrix.shape[1]:
            return
        mat = Matrix.identity()
        assert isinstance(mat, Matrix)

        for index in np.ndindex(mat.shape):
            assert mat[index] == (1.0 + 0.0j if index[0] == index[1] else 0.0j)

    def test_to_numpy(self, Matrix):
        """Test numpy conversion function"""
        mat = Matrix.zeros()
        assert np.allclose(mat.to_numpy(), np.zeros(mat.shape))

    def test_mul(self, Matrix):
        """Test multiplications"""
        mat1_data = np.random.rand(*Matrix.shape)
        mat2_data = np.random.rand(*Matrix.shape)
        mat3_data = np.dot(mat1_data, mat2_data)
        mat1 = Matrix(mat1_data)
        mat2 = Matrix(mat2_data)

        if len(Matrix.shape) > 1 and Matrix.shape[0] == Matrix.shape[1]:
            mat3 = mat1 * mat2
            assert np.allclose(mat3.to_numpy(), mat3_data)
        mat3_data = mat1_data * 5.0
        mat3 = mat1 * 5.0
        assert np.allclose(mat3.to_numpy(), mat3_data)
        mat3_data = mat1_data * (5.0 + 1.0j)
        mat3 = mat1 * (5.0 + 1.0j)
        assert np.allclose(mat3.to_numpy(), mat3_data)
        mat3 = (5.0 + 1.0j) * mat1
        assert np.allclose(mat3.to_numpy(), mat3_data)

    def test_div(self, Matrix):
        """Test division"""
        # TODO: Implement

    def test_add(self, Matrix):
        """Test addition"""
        # TODO: Implement

    def test_sub(self, Matrix):
        """Test subtraction"""
        # TODO: Implement