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
