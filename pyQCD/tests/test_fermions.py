from __future__ import absolute_import

import numpy as np
import pytest

from pyQCD import core, fermions, gauge


@pytest.fixture
def layout():
    return core.LexicoLayout([8, 4, 4, 4])


class TestFermionAction(object):

    def test_constructor(self):
        """Test constructor of FermionAction object"""
        with pytest.raises(NotImplementedError):
            action = fermions.FermionAction()


class TestWilsonFermionAction(object):

    def test_constructor(self, layout):
        """Test construction of WilsonFermionAction"""
        gauge_field = core.LatticeColourMatrix(layout, 4)

        action = fermions.WilsonFermionAction(0.1, gauge_field, [0] * 4)

    def test_apply_full(self, layout):
        """Test full application of WilsonFermionAction"""
        gauge_field = core.LatticeColourMatrix(layout, 4)

        random_gauge_link = core.ColourMatrix.random()
        gauge_field.as_numpy[0, 0, 0, 0, 0] = random_gauge_link.as_numpy

        fermion_in = core.LatticeColourVector(layout, 4)
        fermion_out = core.LatticeColourVector(layout, 4)
        fermion_in.as_numpy[1, 0, 0, 0, 2, :] = 1.0

        action = fermions.WilsonFermionAction(0.1, gauge_field, [0] * 4)
        action.apply_full(fermion_out, fermion_in)

        expected = 0.5 * np.dot(random_gauge_link.as_numpy, np.ones(3))

        assert np.allclose(fermion_out.as_numpy[0, 0, 0, 0, 0], expected)