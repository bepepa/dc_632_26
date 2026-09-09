import numpy as np
import numpy.typing as npt
from itertools import batched

def bits_to_int(bits: npt.ArrayLike) -> int:
    """
    Function that takes a list of bit represented by integers
    and outputs an integer.

    Parameters
    ----------
    bits : npt.ArrayLike
        bitstring represented as zeros and ones.

    Returns
    -------
    int
        The bits converted to an integer
    """

    result = 0
    bits = np.asarray(bits)
    # MSB to LSB
    for bit in bits:
        result = result << 1
        result += bit
    return result

def modulation(bits: np.ndarray, constellation: dict[int, np.complex128]) -> np.ndarray:
    """
    Function that takes bits and maps them according to the mapper.
    The bits are a numpy ndarray, the mapper is a function that should
    take the bits two at a time and map them to a complex value.

    Parameters
    ----------
    bits : np.ndarray
        bitstring represented as zeros and ones
    constellation : dict[int, np.complex128]
        mapping between values of the bits, in integer form,
        and the actual complex value of the symbol sequence at
        that point.

    Returns
    -------
    np.ndarray
        The complex symbols corresponding to the given
        bitstring
    """

    bits_per_symbol = int(np.log2(len(constellation)))
    remainder = bits.size % bits_per_symbol
    if remainder != 0:
        padding = bits_per_symbol - remainder
        bits = np.append(bits, [0]*padding).astype('int')
    symbols = np.empty(bits.size//bits_per_symbol, dtype=np.complex128)
    for idx, bit_seq in enumerate(batched(bits, bits_per_symbol)):
       symbols[idx] = constellation[bits_to_int(bit_seq)]

    return symbols
