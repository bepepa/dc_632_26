# this module contains utility functions for a digital communication system
"""
Utility module for a digital communication system.

This module contains utility functions for our digital signals.

"""

import numpy as np
import numpy.typing as npt


@staticmethod
def Qfunction(x):
    """
    Q function
    """
    from scipy.special import erfc

    return 0.5 * erfc(x / np.sqrt(2))


def bytes_to_bits(byte_seq):
    """Convert a sequence of bytes into a sequence of bits (MSB first)

    Input:
    ------
    byte_seq: a sequence of bytes

    Returns:
    --------
    a vector of binary values, 8 times as long as the inout sequence (stored as uint8)
    """
    # make space for results
    Nb = len(byte_seq)
    bits = np.zeros(8 * Nb, dtype=np.uint8)

    for n in range(Nb):
        bits[8 * n : 8 * (n + 1)] = byte_to_bits(byte_seq[n])

    return bits


def bits_to_bytes(bit_seq):
    """Convert a sequence of bits into a byte sequence (MSB first)

    Input:
    ------
    * bit_seq: a sequence of binary values

    Returns:
    --------
    a vector of bytes (uint8), length is 1/8 of input length

    Exception:
    ----------
    A `ValueError` is raised if length of bit sequence is not divisible by 8.
    """
    Nb = len(bit_seq)
    if (Nb % 8) != 0:
        raise ValueError("number of bits must be a multiple of 8")

    byte_seq = np.zeros(Nb // 8, dtype=np.uint8)

    for n in range(Nb // 8):
        byte_seq[n] = bits_to_byte(bit_seq[8 * n : 8 * (n + 1)])

    return byte_seq


#
# bit-wrangling functions
#
def byte_to_bits(b):
    """convert a single byte to a sequence of 8 bits (MSB first)

    Inputs:
    -------
    b: a single byte

    Returns:
    --------
    a NumPy vector of bits, stored as uint8
    """

    # allocate memory for bits
    bits = np.zeros(8, dtype=np.uint8)

    # define the mask
    mask = 128

    for n in range(8):
        # extract the MSB and store it
        bits[n] = (b & mask) >> 7
        # shift the bits by one position
        b = b << 1

    return bits


def bits_to_byte(bits):
    """convert a sequence of up to eight bits to a byte (MSB in first position)

    Inputs:
    -------
    bits: an iterable of eight 0's and 1's

    Returns:
    --------
    (uint8) a single byte

    """
    assert len(bits) <= 8, "Can only convert at most 8 bits at a time"

    res = np.uint8(0)

    for b in bits:
        res = res << 1
        res = res | b

    return res


def int_to_bits(int_N: npt.ArrayLike, bit_len: int) -> np.ndarray:
    """Convert array of integers to array of bits with specified bit-length"""

    bit_positions = np.arange(bit_len - 1, -1, -1)  # MSB to LSB: [bit_len-1, ..., 0]
    int_reshaped = np.asarray(int_N)[
        :, None
    ]  # Reshape for broadcasting over all integers
    bits_array = (
        int_reshaped >> bit_positions
    ) & 1  # Extract bit at each position for each integer
    return bits_array.ravel()  # Flatten to 1D bitstream


def crc16(data: bytes | bytearray, poly=0x8408):
    """
    CRC-16-CCITT Algorithm

    Args:
        data (bytes): byte sequence to be checked (type bytes)
        poly (int): 16-bit integer indicating the polynomial used by CRC (default 0x8408)

    Returns:
        16-bit CRC
    """
    data = bytearray(data)
    crc = 0xFFFF
    for b in data:
        cur_byte = 0xFF & b
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1
    crc = ~crc & 0xFFFF
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    return crc & 0xFFFF
