import numpy as np
import pytest

from dc_632_26.constellations import BPSK, PSK8, QAM16, QPSK, Constellation


class TestGetItem:
    """Tests __getitem__."""

    def test_bpsk_indexing(self):
        """Tests __getitem__ BPSK."""
        assert BPSK[0] == pytest.approx(1 + 0j)
        assert BPSK[1] == pytest.approx(-1 + 0j)

    def test_qpsk_indexing(self):
        """Tests __getitem__ BPSK."""
        assert QPSK[0] == pytest.approx(1 + 1j)
        assert QPSK[3] == pytest.approx(1 - 1j)


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
