import pytest
import numpy as np
import numpy.testing as test
from dc_632_26.modulation import modulation
from dc_632_26.demodulation import GenericSlicer
from dc_632_26.constellations import QPSK, QAM16
from dc_632_26.utils import int_to_bits


# QPSK constellation: [1+1j, -1+1j, 1-1j, -1-1j]
# Array indices:      [0,    1,     2,    3    ]
# Binary mapping:     [0b00, 0b01,  0b10, 0b11]

@pytest.fixture
def qpsk_slicer():
    return GenericSlicer(QPSK)

@pytest.fixture
def qam16_slicer():
    return GenericSlicer(QAM16)


@pytest.mark.parametrize("int_array, bit_len, expected_bits", [
    (
        np.array([0]), 
        2, 
        np.array([0, 0])
    ),
    (
        np.array([0, 1, 2, 3]), 
        2, 
        np.array([0,0, 0,1, 1,0, 1,1])
    ),
    (
        np.array([7, 0, 5]), 
        3, 
        np.array([1,1,1, 0,0,0, 1,0,1])
    ),
    (
        np.array([7, 0, 5]), 
        7, 
        np.array([0,0,0,0,1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,1,0,1])
    ),
])
def test_int_to_bits(int_array, bit_len, expected_bits):
    result = int_to_bits(int_array, bit_len)
    test.assert_array_equal(result, expected_bits)

@pytest.mark.parametrize("noisy_symbols, expected_bits", [
    # Empty array
    (
        np.array([]), 
        np.array([])
    ),
    # Single symbol
    (
        np.array([1+1j]), 
        np.array([0, 0])
    ),
    # Multiple exact symbols
    (
        np.array([1+1j, -1+1j, 1-1j, -1-1j]), 
        np.array([0,0, 0,1, 1,0, 1,1])
    ),
    # Multiple noisy symbols
    (
        np.array([1.1+0.9j, -0.9+1.1j]), 
        np.array([0,0, 0,1])
    ),
])
def test_generic_slicer_for_qpsk(qpsk_slicer, noisy_symbols, expected_bits):
    result = qpsk_slicer.slice_symbols(noisy_symbols)
    test.assert_array_equal(result, expected_bits)

@pytest.mark.parametrize("noisy_symbols, expected_bits", [
    # Empty array
    (
        np.array([]), 
        np.array([])
    ),
    # Single symbol
    (
        np.array([1+1j]), 
        np.array([1,1,1,1])
    ),
    # Multiple exact symbols
    (
        np.array([1+1j, -1+1j, 1-1j, 3+3j]), 
        np.array([1,1,1,1, 0,1,1,1, 1,1,0,1, 1,0,1,0])
    ),
    # Multiple noisy symbols
    (
        np.array([1.1+0.9j, -0.9+1.1j]), 
        np.array([1,1,1,1, 0,1,1,1])
    ),
])
def test_generic_slicer_for_qam16(qam16_slicer, noisy_symbols, expected_bits):
    result = qam16_slicer.slice_symbols(noisy_symbols)
    test.assert_array_equal(result, expected_bits)

def test_demodulation_large_array(qpsk_slicer):
    """Test demodulation with large array - modulate then demodulate should recover bits"""

    # Generate random bit sequence
    np.random.seed(42)
    num_bits = 10001
    original_bits = np.random.randint(0, 2, size=num_bits)
    
    # Modulate to symbols
    symbols = modulation(original_bits, QPSK)
    
    # Demodulate back to bits
    recovered_bits = qpsk_slicer.slice_symbols(symbols)
    
    # Should match original, remove padding at end
    test.assert_array_equal(recovered_bits[:num_bits], original_bits)

def test_demodulation_large_array(qam16_slicer):
    """Test demodulation with large array - modulate then demodulate should recover bits"""

    # Generate random bit sequence
    np.random.seed(42)
    num_bits = 10001
    original_bits = np.random.randint(0, 2, size=num_bits)
    
    # Modulate to symbols
    symbols = modulation(original_bits, QAM16)
    
    # Demodulate back to bits
    recovered_bits = qam16_slicer.slice_symbols(symbols)
    
    # Should match original, remove padding at end
    test.assert_array_equal(recovered_bits[:num_bits], original_bits)