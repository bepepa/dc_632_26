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
        result = constellation.get_number_bits_per_symbol()
        assert result == expected_bits


class TestSymbolEnregy:
    """Tests for get_symbol_energy."""

    def test_bpsk_symbol_energy(self):
        """test_bpsk_symbol_energy."""
        assert BPSK.get_symbol_energy() == pytest.approx(1.0)

    def test_qpsk_symbol_energy(self):
        """test_qpsk_symbol_energy."""
        assert QPSK.get_symbol_energy() == pytest.approx(2.0)

    def test_psk8_symbol_energy(self):
        """test_psk8_symbol_energy."""
        # on unit circle
        assert PSK8.get_symbol_energy() == pytest.approx(1.0)

    def test_qam16_symbol_energy(self):
        """test_qam16_symbol_energy."""
        assert QAM16.get_symbol_energy() == pytest.approx(10.0)