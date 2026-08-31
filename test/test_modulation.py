import numpy as np
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
    print(symbols)

    # Test when there is a remainder
    bits = np.array([0,1,1,1,0,1,1])
    symbols = modulation(bits, TEST_QPSK)
    print(symbols)