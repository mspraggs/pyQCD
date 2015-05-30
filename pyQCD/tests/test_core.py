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


class TestColourMatrix(object):

    def test_constructor(self):
        """Test matrix"""
        mat = ColourMatrix()
        assert isinstance(mat, ColourMatrix)
        mat_values = np.arange(np.prod(mat.shape)).reshape(mat.shape).tolist()
        mat = ColourMatrix(mat_values)
        assert isinstance(mat, ColourMatrix)

        for i, j in np.ndindex(mat.shape):
            assert mat[i, j] == mat_values[i][j] + 0j

        mat_values = np.arange(np.prod(mat.shape)).reshape(mat.shape)
        mat = ColourMatrix(mat_values)
        assert isinstance(mat, ColourMatrix)

        for index in np.ndindex(mat.shape):
            assert mat[index] == mat_values[index]

        with pytest.raises(ValueError):
            mat = ColourMatrix(np.zeros((20, 20)))

    def test_boundscheck(self):
        """Test bounds checking for matrix"""
        mat = ColourMatrix()
        with pytest.raises(IndexError):
            x = mat[3, 3]
        with pytest.raises(IndexError):
            mat[3, 3] = 4

    def test_zeros(self):
        """Test zeros static function"""
        mat = ColourMatrix.zeros()
        assert isinstance(mat, ColourMatrix)

        for index in np.ndindex(mat.shape):
            assert mat[index] == 0.0j

    def test_ones(self):
        """Test ones static function"""
        mat = ColourMatrix.ones()
        assert isinstance(mat, ColourMatrix)

        for index in np.ndindex(mat.shape):
            assert mat[index] == 1.0 + 0.0j

    def test_identity(self):
        """Test identity static function"""
        mat = ColourMatrix.identity()
        assert isinstance(mat, ColourMatrix)

        for index in np.ndindex(mat.shape):
            assert mat[index] == (1.0 + 0.0j if index[0] == index[1] else 0.0j)

    def test_to_numpy(self):
        """Test numpy conversion function"""
        mat = ColourMatrix.zeros()
        assert np.allclose(mat.to_numpy(), np.zeros((3, 3)))

    def test_mul(self):
        """Test multiplications"""
        mat1_data = np.random.rand(3, 3)
        mat2_data = np.random.rand(3, 3)
        mat3_data = np.dot(mat1_data, mat2_data)
        mat1 = ColourMatrix(mat1_data)
        mat2 = ColourMatrix(mat2_data)

        mat3 = mat1 * mat2
        assert np.allclose(mat3.to_numpy(), mat3_data)
        mat3_data = mat1_data * 5.0
        mat3 = mat1 * 5.0
        assert np.allclose(mat3.to_numpy(), mat3_data)
        mat3_data = mat1_data * (5.0 + 1.0j)
        mat3 = mat1 * (5.0 + 1.0j)
        assert np.allclose(mat3.to_numpy(), mat3_data)


class TestColourVector(object):

    def test_constructor(self):
        """Test vector constructor"""
        vec = ColourVector()
        assert isinstance(vec, ColourVector)
        vec_values = np.arange(np.prod(vec.shape)).reshape(vec.shape).tolist()
        vec = ColourVector(vec_values)
        assert isinstance(vec, ColourVector)

        for i, in np.ndindex(vec.shape):
            assert vec[i] == vec_values[i] + 0j

        vec_values = np.arange(np.prod(vec.shape)).reshape(vec.shape)
        vec = ColourVector(vec_values)
        assert isinstance(vec, ColourVector)

        for index in np.ndindex(vec.shape):
            assert vec[index] == vec_values[index]

        with pytest.raises(ValueError):
            vec = ColourVector(np.zeros(20))

    def test_boundscheck(self):
        """Test bounds checking for matrix"""
        vec = ColourVector()
        with pytest.raises(IndexError):
            x = vec[3]
        with pytest.raises(IndexError):
            vec[3] = 4

    def test_zeros(self):
        """Test zeros static function"""
        vec = ColourVector.zeros()
        assert isinstance(vec, ColourVector)

        for index in np.ndindex(vec.shape):
            assert vec[index] == 0.0j

    def test_ones(self):
        """Test ones static function"""
        vec = ColourVector.ones()
        assert isinstance(vec, ColourVector)

        for index in np.ndindex(vec.shape):
            assert vec[index] == 1.0 + 0.0j