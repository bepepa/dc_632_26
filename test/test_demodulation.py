import pytest
import numpy as np
import numpy.testing as test
from dc_632_26.modulation import modulation
from dc_632_26.demodulation import GenericSlicer, int_to_bits, demodulation
from dc_632_26.constellations import QPSK

# QPSK constellation: [1+1j, -1+1j, 1-1j, -1-1j]
# Array indices:      [0,    1,     2,    3    ]
# Binary mapping:     [0b00, 0b01,  0b10, 0b11]

@pytest.fixture
def qpsk_slicer():
    return GenericSlicer(QPSK)

@pytest.mark.parametrize("noisy_symbols, expected_indices", [
    (
        np.array([1+1j, -1+1j, 1-1j, -1-1j]), 
        np.array([0, 1, 2, 3])
    ),
    (
        np.array([1.1+0.9j, -0.9+1.1j, 1.05-0.95j]), 
        np.array([0, 1, 2])
    ),
    (
        np.array([0.01+0.5j, -0.01-0.5j]), 
        np.array([0, 3])
    ),
    (
        np.array([100+100j, -1000+500j, 5000-3000j]), 
        np.array([0, 1, 2])
    ),
])
def test_generic_slicer(qpsk_slicer, noisy_symbols, expected_indices):
    result = qpsk_slicer.slice_symbols(noisy_symbols)
    test.assert_array_equal(result, expected_indices)

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
def test_demodulation(qpsk_slicer, noisy_symbols, expected_bits):
    result = demodulation(noisy_symbols, qpsk_slicer)
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
    recovered_bits = demodulation(symbols, qpsk_slicer)
    
    # Should match original, remove padding at end
    test.assert_array_equal(recovered_bits[:num_bits], original_bits)