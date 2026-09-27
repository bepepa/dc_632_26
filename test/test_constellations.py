'''
Test suite for Constellation class.

Run tests:
    pytest tests/test_constellations.py                    # Run all tests in this file
    pytest tests/test_constellations.py -v                 # Verbose output
    pytest tests/test_constellations.py::TestBitEnergy     # Run specific test class
    pytest -k "bpsk"                                       # Run tests matching pattern
    pytest tests/test_constellations.py --tb=short         # Shorter traceback

Note, direct imports of constellations example: constellations.BPSK is legacy useage. 
'''

import numpy as np
import pytest

from dc_632_26.constellations import BPSK, PSK8, QAM16, QPSK, APSK16 , APSK_constellation   # legacy imports
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
        assert standard_constellation.QPSK[3] == pytest.approx(-1/np.sqrt(2) - 1j/np.sqrt(2))


###defining R1 and R2 for APSK16 tests
R1 = 1.0
R2 = R1 * 3.15  # outer ring radius scaled by gamma factor       
@pytest.mark.parametrize(
    "constellation, expected_points",
    [
        (BPSK, [1 + 0j, -1 + 0j]),
        (
            QPSK,
            [
                1 + 1j,
                -1 + 1j,
                1 - 1j,
                -1 - 1j,
            ],
        ),
        (
            PSK8,
            [
                1 + 0j,
                np.sqrt(2) / 2 * (1 + 1j),
                np.sqrt(2) / 2 * (-1 + 1j),
                0 + 1j,
                np.sqrt(2) / 2 * (1 - 1j),
                0 - 1j,
                -1 + 0j,
                np.sqrt(2) / 2 * (-1 - 1j),
            ],
        ),
        (
            QAM16,
            [
                -3 - 3j,
                -3 - 1j,
                -3 + 3j,
                -3 + 1j,
                -1 - 3j,
                -1 - 1j,
                -1 + 3j,
                -1 + 1j,
                3 - 3j,
                3 - 1j,
                3 + 3j,
                3 + 1j,
                1 - 3j,
                1 - 1j,
                1 + 3j,
                1 + 1j,
            ],
        ),

        (APSK16,[    
                # 0000
                R2 * np.exp(1j * np.pi / 4),

                # 0001
                R2 * np.exp(1j * 7 * np.pi / 4),

                # 0010
                R2 * np.exp(1j * 3 * np.pi / 4),

                # 0011
                R2 * np.exp(1j * 5 * np.pi / 4),

                # 0100
                R2 * np.exp(1j * np.pi / 12),

                # 0101
                R2 * np.exp(1j * 23 * np.pi / 12),

                # 0110
                R2 * np.exp(1j * 11 * np.pi / 12),

                # 0111
                R2 * np.exp(1j * 13 * np.pi / 12),

                # 1000
                R2 * np.exp(1j * 5 * np.pi / 12),

                # 1001
                R2 * np.exp(1j * 19 * np.pi / 12),

                # 1010
                R2 * np.exp(1j * 7 * np.pi / 12),

                # 1011
                R2 * np.exp(1j * 17 * np.pi / 12),

                # 1100
                R1 * np.exp(1j * np.pi / 4),

                # 1101
                R1 * np.exp(1j * 7 * np.pi / 4),

                # 1110
                R1 * np.exp(1j * 3 * np.pi / 4),

                # 1111
                R1 * np.exp(1j * 5 * np.pi / 4),
        ]
         ),
    ],
)

class TestLegacyGetItem:
    """Tests __getitem__ for every point in the legacy constellations."""

    def test_get_item(self, constellation, expected_points):
        """Returns the expected constellation point for every index."""
        for i, expected in enumerate(expected_points):
            assert constellation[i] == pytest.approx(expected)
            assert isinstance(constellation[i], complex)

@pytest.mark.parametrize(
    "constellation, expected_bits",
    [
        (standard_constellation.BPSK, 1),
        (standard_constellation.QPSK, 2),
        (standard_constellation.PSK8, 3),
        (standard_constellation.QAM16, 4),
        (standard_constellation.APSK16, 4)
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
        (standard_constellation.APSK16, 1.0),
        (BPSK, 1),
        (QPSK, 2),
        (PSK8, 1),
        (QAM16, 10),
        (APSK16, (4 * R1**2 + 12 * R2**2) / 16),
    ],
)
class TestSymbolEnergy:
    """Tests for get_symbol_energy."""

    def test_symbol_energy(self, constellation, expected_symbol_energy):
        """Run the same test for all parameterized inputs."""
        assert constellation.symbol_energy == pytest.approx(expected_symbol_energy)
        assert constellation.Es == pytest.approx(expected_symbol_energy)

@pytest.mark.parametrize(
    "constellation, expected_bit_energy",
    [
        (standard_constellation.BPSK, 1.0),
        (standard_constellation.QPSK, 0.5),
        (standard_constellation.PSK8, 1.0/3),
        (standard_constellation.QAM16, 0.25),
        (standard_constellation.APSK16, 0.25),
    ],
)
class TestBitEnergy:
    """Tests for bit energy."""

    def test_bit_energy(self, constellation, expected_bit_energy):
        """Test Eb = Es/bps for normalized constellations."""
        assert constellation.bit_energy == pytest.approx(expected_bit_energy)
        assert constellation.Eb == pytest.approx(expected_bit_energy)

@pytest.mark.parametrize(
    "constellation, expected_min_distance",
    [
        (standard_constellation.BPSK, 2.0),
        (standard_constellation.QPSK, np.sqrt(2)),
        (standard_constellation.PSK8, 2 * np.sin(np.pi/8)),
        (standard_constellation.QAM16, 2/np.sqrt(10)),
        (BPSK, 2.0),
        (QPSK, 2.0),
        (PSK8, 2 * np.sin(np.pi/8)),
        (QAM16, 2.0),
    ],
)
class TestMinDistance:
    """Tests for minimum distance."""

    def test_min_distance(self, constellation, expected_min_distance):
        """Test dmin for normalized constellations."""
        assert constellation.min_distance == pytest.approx(expected_min_distance)
        assert constellation.dmin == pytest.approx(expected_min_distance)

@pytest.mark.parametrize(
    "constellation, expected_eta",
    [
        (standard_constellation.BPSK, 4.0),
        (standard_constellation.QPSK, 4.0),
        (standard_constellation.PSK8, (2 * np.sin(np.pi/8))**2 / (1.0/3)),
        (standard_constellation.QAM16, (2/np.sqrt(10))**2 / 0.25),
    ],
)
class TestEnergyEfficiency:
    """Tests for energy efficiency."""

    def test_energy_efficiency(self, constellation, expected_eta):
        """Test eta = dmin²/Eb."""
        assert constellation.energy_efficiency == pytest.approx(expected_eta)
        assert constellation.eta == pytest.approx(expected_eta)

class TestPointDistance:
    """Tests for point_distance method."""

    def test_qpsk_adjacent_points(self):
        """Test distance between adjacent QPSK points."""
        dist = standard_constellation.QPSK.point_distance(0b00, 0b01)
        assert dist == pytest.approx(np.sqrt(2))

    def test_bpsk_points(self):
        """Test distance between BPSK points."""
        dist = standard_constellation.BPSK.point_distance(0b0, 0b1)
        assert dist == pytest.approx(2.0)

@pytest.mark.parametrize(
    "constellation",
    [standard_constellation.BPSK, 
     standard_constellation.QPSK, 
     standard_constellation.PSK8, 
     standard_constellation.QAM16],
)
class TestGrayCoding:
    """Tests for gray coding of standard constellations."""
    
    @staticmethod
    def is_gray_coded(constellation: Constellation) -> bool:
        """Test if constellation is gray-coded.
        
        Returns True if all nearest neighbor pairs differ by exactly 1 bit.
        """
        import numpy as np
        
        symbols = constellation.mod_table
        bit_patterns = np.arange(len(symbols))
        
        # Compute all pairwise distances via broadcasting
        distances = np.abs(symbols[:, np.newaxis] - symbols[np.newaxis, :])
        np.fill_diagonal(distances, np.inf)
        min_dist = distances.min()
        
        # Find nearest neighbor pairs (upper triangle only to avoid duplicates)
        is_nearest = np.abs(distances - min_dist) < 1e-9
        i_indices, j_indices = np.where(np.triu(is_nearest, k=1))
        
        # Check if nearest neighbors differ by exactly 1 bit
        for i, j in zip(i_indices, j_indices):
            bit_diff = bin(bit_patterns[i] ^ bit_patterns[j]).count('1')
            if bit_diff != 1:
                return False
        
        return True
    
    def test_gray_coded(self, constellation):
        """Test constellation is gray-coded."""
        assert self.is_gray_coded(constellation)
