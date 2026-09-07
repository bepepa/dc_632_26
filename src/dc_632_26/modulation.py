import numpy as np
from itertools import batched

def _bits_to_int(bits: np.ndarray):
    counter = 0
    sum = 0
    for bit in bits[::-1]:
        sum += bit*2**counter
        counter += 1
    return sum

def modulation(bits, constellation):
    """
    Function that takes bits and maps them according to the mapper.
    The bits are a numpy ndarray, the mapper is a function that should
    take the bits two at a time and map them to a complex value.

    It returns a complex array.

    TODO: be able to take varying bits per symbol.
    Include support functions for Eb, Es, Bps, eta(energy efficiency)
    Split this out into a function for the constellation and a function
    to actually map bits to symbols given a constellation
    """

    num_bits_to_take = int(np.log2(len(constellation)))
    remainder = bits.size % num_bits_to_take
    if remainder != 0:
        padding = num_bits_to_take - remainder
        bits = np.append(bits, [0]*padding).astype('int')
    # padded_bits = np.append(bits, [0,0]) if bits.size % 2 == 0 else np.append(bits,[0,0,1])
    symbols = np.empty(bits.size//num_bits_to_take, dtype=np.complex128)
    for idx, bit_seq in enumerate(batched(bits, num_bits_to_take)):
       symbols[idx] = constellation[_bits_to_int(bit_seq)]
    # groups = [bits[i:i + num_bits_to_take] for i in range(0, len(bits), num_bits_to_take)]
    # symbols = np.array([constellation[_bits_to_int(group)] for group in groups], dtype=np.complex128)

    return symbols

