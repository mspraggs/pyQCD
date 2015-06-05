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
            x = mat[mat.shape]
        with pytest.raises(IndexError):
            mat[mat.shape] = 4

    def test_adjoint(self, Matrix):
        """Test adjoint function for matrix"""
        mat = Matrix()
        index = tuple(0 for i in mat.shape)
        mat[index] = 1 + 1j
        mat_adjoint = mat.adjoint()
        assert mat_adjoint[index] == 1 - 1j

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

    def test_buffer_protocol(self, Matrix):
        """Test buffer protocol implementation"""
        mat = Matrix.zeros()
        np_mat = np.asarray(mat)
        np_mat.dtype = complex
        for index in np.ndindex(mat.shape):
            assert np_mat[index] == 0j

        index = tuple(s - 1 if i == 0 else 0 for i, s in enumerate(mat.shape))
        np_mat[index] = 5.0
        assert np_mat[index] == 5.0 + 0j
        assert mat[index] == 5.0 + 0j

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
        mat1_data = np.random.rand(*Matrix.shape)
        mat2_data = mat1_data / (4.0 - 2.0j)
        mat1 = Matrix(mat1_data)
        mat2 = mat1 / (4.0 - 2.0j)
        assert np.allclose(mat2.to_numpy(), mat2_data)

    def test_add(self, Matrix):
        """Test addition"""
        mat1_data = np.random.rand(*Matrix.shape)
        mat2_data = np.random.rand(*Matrix.shape)
        mat3_data = mat1_data + mat2_data
        mat1 = Matrix(mat1_data)
        mat2 = Matrix(mat2_data)
        mat3 = mat1 + mat2
        assert np.allclose(mat3.to_numpy(), mat3_data)

    def test_sub(self, Matrix):
        """Test subtraction"""
        mat1_data = np.random.rand(*Matrix.shape)
        mat2_data = np.random.rand(*Matrix.shape)
        mat3_data = mat1_data - mat2_data
        mat1 = Matrix(mat1_data)
        mat2 = Matrix(mat2_data)
        mat3 = mat1 - mat2
        assert np.allclose(mat3.to_numpy(), mat3_data)


@pytest.mark.parametrize("MatrixArray,Matrix",
                         [(ColourMatrixArray, ColourMatrix),
                          (Fermion, ColourVector)])
class TestMatrixArrayType(object):

    def test_constructor(self, MatrixArray, Matrix):
        """Test constructor"""
        mat_arr = MatrixArray()
        assert isinstance(mat_arr, MatrixArray)
        mat_arr = MatrixArray(4, Matrix.zeros())
        assert isinstance(mat_arr, MatrixArray)
        assert mat_arr.size == 4
        assert mat_arr.shape == (4,) + Matrix.shape

        mat_arr_data = np.arange(np.prod(mat_arr.shape)).reshape(mat_arr.shape)
        mat_arr = MatrixArray(mat_arr_data.tolist())
        assert isinstance(mat_arr, MatrixArray)
        for index in np.ndindex(mat_arr.shape):
            assert mat_arr[index[0]][index[1:]] == mat_arr_data[index]

        mat_arr = MatrixArray(mat_arr_data)
        assert isinstance(mat_arr, MatrixArray)
        for index in np.ndindex(mat_arr.shape):
            assert mat_arr[index[0]][index[1:]] == mat_arr_data[index]

    def test_boundscheck(self, Matrix, MatrixArray):
        """Test bounds checking for matrix array type"""
        mat = Matrix.zeros()
        mat_arr = MatrixArray(4, mat)
        with pytest.raises(IndexError):
            x = mat_arr[4]
        with pytest.raises(IndexError):
            x = mat_arr[mat_arr.shape]
        with pytest.raises(IndexError):
            mat_arr[4] = mat
        with pytest.raises(IndexError):
            mat_arr[mat_arr.shape] = 4

    def test_adjoint(self, Matrix, MatrixArray):
        """Test adjoint function for matrix"""
        mat = Matrix()
        mat_arr = MatrixArray(4, mat)
        index = tuple(0 for i in mat_arr.shape)
        mat_arr[index] = 1 + 1j
        mat_arr_adjoint = mat_arr.adjoint()
        assert mat_arr_adjoint[index] == 1.0 - 1.0j

    def test_zeros(self, Matrix, MatrixArray):
        """Test zeros static function"""
        mat_arr = MatrixArray.zeros(4)
        assert isinstance(mat_arr, MatrixArray)
        assert mat_arr.size == 4
        assert mat_arr.shape == (4,) + Matrix.shape

        for index in np.ndindex(mat_arr.shape):
            assert mat_arr[index] == 0.0j

    def test_ones(self, Matrix, MatrixArray):
        """Test ones static function"""
        mat_arr = MatrixArray.ones(4)
        assert isinstance(mat_arr, MatrixArray)
        assert mat_arr.size == 4
        assert mat_arr.shape == (4,) + Matrix.shape

        for index in np.ndindex(mat_arr.shape):
            assert mat_arr[index] == 1.0 + 0.0j

    def test_identity(self, Matrix, MatrixArray):
        """Test identity static function"""
        if len(Matrix.shape) == 1 or Matrix.shape[0] != Matrix.shape[1]:
            return
        mat_arr = MatrixArray.identity(4)
        assert isinstance(mat_arr, MatrixArray)
        assert mat_arr.size == 4
        assert mat_arr.shape == (4,) + Matrix.shape

        for index in np.ndindex(mat_arr.shape):
            assert mat_arr[index] == (1.0 + 0.0j if index[1] == index[2] else 0.0j)

    def test_to_numpy(self, Matrix, MatrixArray):
        """Test numpy conversion function"""
        mat_arr = MatrixArray.zeros(4)
        assert np.allclose(mat_arr.to_numpy(), np.zeros(mat_arr.shape))

    def test_buffer_protocol(self, Matrix, MatrixArray):
        """Test buffer protocol implementation"""
        mat_arr = MatrixArray.zeros(4)
        np_mat = np.asarray(mat_arr)
        np_mat.dtype = complex
        for index in np.ndindex(mat_arr.shape):
            assert np_mat[index] == 0j

        index = tuple(s - 1 if i == 0 else 0
                      for i, s in enumerate(mat_arr.shape))
        np_mat[index] = 5.0
        assert np_mat[index] == 5.0 + 0j
        assert mat_arr[index] == 5.0 + 0j
