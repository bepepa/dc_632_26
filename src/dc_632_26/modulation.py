import numpy as np
import numpy.typing as npt
from dc_632_26.constellations import *

def bits_to_int( bits_N: npt.ArrayLike, bits_per_symbol: int ) -> np.ndarray:
    """
    Function that takes a list of bit represented by integers
    and outputs an integer.

    Parameters
    ----------
    bits_BS : npt.ArrayLike
        bitstring represented as zeros and ones.
        Shaped to process multiple symbols at once.
        Number of rows is number of bits per symbol. MSB is first(top) - B
        Number of columns is number of symbols. - S
    Returns
    -------
    np.ndarray
        The bits converted to an integer
    """

    bits_N = np.asarray( bits_N )
    bits_BS = np.reshape( bits_N, (bits_per_symbol, -1),order='F')
    print(bits_BS)
    result_S = np.zeros( bits_BS.shape[-1], dtype='int')
    # MSB to LSB
    for bit in bits_BS:
        result_S = result_S << 1
        result_S += bit
    return result_S

def modulation( bits_N: np.ndarray, constellation_map_O: Constellation ) -> np.ndarray:
    """
    Function that takes bits and maps them according to the mapper.
    The bits are a numpy ndarray, the mapper is a function that should
    take the bits bits_per_symbol at a time and map them to a complex value.

    Parameters
    ----------
    bits_N : np.ndarray
        bitstring represented as zeros and ones
        N is the number of bits, it can change slightly
        if we pad out the sequence.
    constellation_map_O : np.ndarray
        mapping between values of the bits, as indices,
        and the actual complex value of the symbol sequence at
        that point.
        O is the modulation order.

    Returns
    -------
    np.ndarray
        The complex symbols corresponding to the given
        bitstring
        S is the number of symbols
        B is the number of bits per symbol
    """

    bits_per_symbol = int( np.log2( len( constellation_map_O() ) ) )
    remainder = bits_N.size % bits_per_symbol
    if remainder != 0:
        padding = bits_per_symbol - remainder
        bits_N = np.append( bits_N, [0] * padding).astype('int')
    symbols_S = constellation_map_O()[ bits_to_int( bits_N, bits_per_symbol ) ]

    return symbols_S
