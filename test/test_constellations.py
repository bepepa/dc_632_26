import numpy as np
import pytest

from dc_632_26.constellations import standard_constellation, Constellation


class TestGetItem:
    """Tests __getitem__."""

    def test_bpsk_indexing(self):
        """Tests __getitem__ BPSK."""
        assert standard_constellation.BPSK[0] == pytest.approx(1 + 0j)
        assert standard_constellation.BPSK[1] == pytest.approx(-1 + 0j)

    def test_qpsk_indexing(self):
        """Tests __getitem__ QPSK."""
        # QPSK is normalized, so divided by sqrt(2)
        assert standard_constellation.QPSK[0] == pytest.approx(1/np.sqrt(2) + 1j/np.sqrt(2))
        assert standard_constellation.QPSK[3] == pytest.approx(1/np.sqrt(2) - 1j/np.sqrt(2))


@pytest.mark.parametrize(
    "constellation, expected_bits",
    [
        (standard_constellation.BPSK, 1),
        (standard_constellation.QPSK, 2),
        (standard_constellation.PSK8, 3),
        (standard_constellation.QAM16, 4),
    ],
)
class TestNumberOfBitsPerSymbol:
    """Tests for getting the number of bits in predefined constellations."""

    def test_bits_per_symbol(self, constellation, expected_bits):
        """Run the same test for all parameterized inputs."""
        assert constellation.bits_per_symbol == expected_bits
        assert constellation.bps == expected_bits


@pytest.mark.parametrize(
    "constellation, expected_symbol_energy",
    [
        (standard_constellation.BPSK, 1.0),
        (standard_constellation.QPSK, 1.0),
        (standard_constellation.PSK8, 1.0),
        (standard_constellation.QAM16, 1.0),
    ],
)
class TestSymbolEnergy:
    """Tests for get_symbol_energy."""

    def test_symbol_energy(self, constellation, expected_symbol_energy):
        """Run the same test for all parameterized inputs."""
        assert constellation.symbol_energy == pytest.approx(expected_symbol_energy)
        assert constellation.Es == pytest.approx(expected_symbol_energy)