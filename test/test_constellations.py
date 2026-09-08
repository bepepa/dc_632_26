"""
Completes Bulding tests for the support functions
developed within constellations.py constellations class. 
"""

import numpy as np
import pytest

from dc_632_26.constellations import BPSK, PSK8, QAM16, QPSK, Constellation

"""
__getitem__ loops over every point of all four constellations.
"""
@pytest.mark.parametrize(
    "constellation, expected_points",
    [
        (BPSK, [1 + 0j, -1 + 0j]),
        (QPSK, [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]),
        (PSK8, [1 + 0j, np.sqrt(2) / 2 * (1 + 1j),
                0 + 1j, np.sqrt(2) / 2 * (-1 + 1j),
                -1 + 0j, np.sqrt(2) / 2 * (-1 - 1j),
                0 - 1j, np.sqrt(2) / 2 * (1 - 1j)]),
        (QAM16, [-3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
                 -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
                 3 - 3j, 3 - 1j, 3 + 3j, 3 + 1j,
                 1 - 3j, 1 - 1j, 1 + 3j, 1 + 1j])
    ]
)

class TestGetItem:
    """Tests __getitem__."""

    def test_get_item(self, constellation, expected_points):
        """returns expected constellation point for every index i"""
        for i, expected in enumerate(expected_points):
            assert constellation[i] == pytest.approx(expected)
            assert isinstance(constellation[i], complex)


@pytest.mark.parametrize(
    "constellation, expected_bits",
    [
        (BPSK, 1),
        (QPSK, 2),
        (PSK8, 3),
        (QAM16, 4),
    ],
)

class TestNumberOfBitsPerSymbol:
    """Tests for getting the number of bits in predefine constellations."""

    def test_bits_per_symbol(self, constellation, expected_bits):
        """Run the same test for all parameterized inputs."""
        assert constellation.bits_per_symbol == expected_bits
        assert constellation.bps == expected_bits


@pytest.mark.parametrize(
    "constellation, expected_symbol_energy",
    [
        (BPSK, 1),
        (QPSK, 2),
        (PSK8, 1),
        (QAM16, 10),
    ],
)

class TestSymbolEnregy:
    """Tests for get_symbol_energy."""

    def test_symbol_energy(self, constellation, expected_symbol_energy):
        """Run the same test for all parameterized inputs."""
        assert constellation.symbol_energy == pytest.approx(expected_symbol_energy)
        assert constellation.Es == pytest.approx(expected_symbol_energy)


@pytest.mark.parametrize(
    "constellation, expected_bit_energy",
    [
        (BPSK, 1.0),
        (QPSK, 1.0),
        (PSK8, 1/3),
        (QAM16, 2.5),
    ],
)

class TestBitEnergy:
    """Tests for get_bit_energy."""

    def test_bit_energy(self, constellation, expected_bit_energy):
        """Run the same test for all parametrized inputs."""
        assert constellation.bit_energy == pytest.approx(expected_bit_energy)
        assert constellation.Eb == pytest.approx(expected_bit_energy)


@pytest.mark.parametrize(
    "constellation, expected_min_distance",
    [
        (BPSK, 2),
        (QPSK, 2),
        (PSK8, 2*np.sin(np.pi/8)),
        (QAM16, 2),
    ],
)

class TestMinDistance:
    """Tests for get_min_distance."""

    def test_min_distance(self, constellation, expected_min_distance):
        """Run the same test for all parametrized inputs"""
        assert constellation.min_distance == pytest.approx(expected_min_distance)
        assert constellation.dmin == pytest.approx(expected_min_distance)

    
@pytest.mark.parametrize(
    "constellation, expected_energy_efficiency",
    [
        (BPSK, 4),
        (QPSK, 4),
        (PSK8, (2 * np.sin(np.pi / 8)) ** 2 / (1 / 3)),
        (QAM16, 1.6),
    ],
)

class TestEnergyEfficiency:
    """Tests for get_energy_efficiency."""

    def test_energy_efficiency(self, constellation, expected_energy_efficiency):
        """Run the same test for all parametrized inputs"""
        assert constellation.energy_efficiency == pytest.approx(expected_energy_efficiency)
        assert constellation.eta == pytest.approx(expected_energy_efficiency)