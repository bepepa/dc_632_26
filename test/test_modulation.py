import numpy as np
import numpy.testing as test
from dc_632_26.modulation import *

TEST_QPSK = {
    0: 1+1j,
    1: -1+1j,
    3: -1-1j,
    2: 1-1j
}

def test_modulation():
    # Test base case of no remainder
    bits = np.array([0,1,1,1,0,1,1,0])
    symbols = modulation(bits, TEST_QPSK)
    test.assert_array_equal(symbols,np.array([-1+1j, -1-1j, -1+1j, 1-1j]))

    # Test when there is a remainder
    bits = np.array([0,1,1,1,0,1,1])
    symbols = modulation(bits, TEST_QPSK)
    test.assert_array_equal(symbols,np.array([-1+1j,-1-1j,-1+1j,1-1j]))

def test_bits_to_int():
    # 0
    bits = np.array([0])
    result = bits_to_int(bits)
    test.assert_equal(result, 0)
    # 111
    bits = np.array([1,1,1])
    result = bits_to_int(bits)
    test.assert_equal(result, 7)
    # 0110
    bits = np.array([0,1,1,0])
    result = bits_to_int(bits)
    test.assert_equal(result, 6)
