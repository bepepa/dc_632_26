# this module contains utility functions for a digital communication system
"""
Utility module for a digital communication system.

This module contains utility functions for our digital signals.

"""

import numpy as np

@staticmethod
def Qfunction(x):
    """
    Q function
    """
    from scipy.special import erfc
    return 0.5 * erfc(x / np.sqrt(2))

def int_to_bits(int_N: npt.ArrayLike, bit_len: int) -> np.ndarray:
    """Convert array of integers to array of bits with specified bit-length"""

    bit_positions = np.arange(bit_len - 1, -1, -1)    # MSB to LSB: [bit_len-1, ..., 0]
    int_reshaped = int_N[:, None]                     # Reshape for broadcasting over all integers
    bits_array = (int_reshaped >> bit_positions) & 1  # Extract bit at each position for each integer
    return bits_array.ravel()                         # Flatten to 1D bitstream