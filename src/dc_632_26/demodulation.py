"""Digital baseband demodulation and generic slicer.

This file defines the demodulation function, Slicer abstract base class,
GenericSlicer class implementation for converting noisy received symbols back to bits.

The Slicer class: 
  - Abstract base class defining the interface for all slicer implementations
  - Requires a decision map and decision metric to map noisy symbols to constellation coordinates
  - Slicer implementations must provide a `slice_symbols()` method that returns constellation indices

The demodulation function combines slicing and bit conversion operations: 
  - (1) takes noisy symbols and a slicer object
  - (2) maps symbols to constellation indices
  - (3) extracts the corresponding bit pattern

This design mirrors the modulation function for symmetric transmitter/receiver implementation.

Example
-------
>>> from dc_632_26.constellations import QPSK
>>> from dc_632_26.demodulation import GenericSlicer, demodulation
>>> 
>>> slicer = GenericSlicer(QPSK)
>>> rx_bits = demodulation(noisy_symbols, slicer)
"""

import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from dc_632_26.constellations import Constellation

class Slicer(ABC):
    """Base class for all slicer implementations"""
    @abstractmethod
    def slice_symbols(self, noisy_symbols: np.ndarray) -> np.ndarray:
        pass

class GenericSlicer(Slicer):
    """Generic nearest-neighbor slicer for any decision-map/constellation.

    Accepts a Constellation object as its decision map and uses nearest-neighbor distance 
    as the decision metric to map noisy symbols to the closest constellation coordinate. 

    Parameters
    ----------
    constellation : Constellation
        Constellation object containing symbol mapping
    """
    
    def __init__(self, constellation: Constellation):
        self.constellation = constellation
        self.decision_map_symbols = self.constellation()
        self.bits_per_symbol = self.constellation.bps

    def slice_symbols(self, noisy_symbols: np.ndarray) -> np.ndarray:
        """Map noisy symbols to nearest constellation points using minimum distance."""

        noisy_symbols_reshaped = noisy_symbols[:, np.newaxis]                   # Apply broadcasting: (N,1) - (M,) → (N,M) 
        distances = np.abs(noisy_symbols_reshaped - self.decision_map_symbols)  # compute all pairwise distances
        return np.argmin(distances, axis=1)                                     # Return index, 'm', of nearest constellation point

def int_to_bits(int_N: npt.ArrayLike, bit_len: int) -> np.ndarray:
    """Convert array of integers to array of bits with specified bit-length"""

    bit_positions = np.arange(bit_len - 1, -1, -1)    # MSB to LSB: [bit_len-1, ..., 0]
    int_reshaped = int_N[:, None]                     # Reshape for broadcasting over all integers
    bits_array = (int_reshaped >> bit_positions) & 1  # Extract bit at each position for each integer
    return bits_array.ravel()                         # Flatten to 1D bitstream

def demodulation(noisy_symbols: np.ndarray, slicer: Slicer) -> np.ndarray:
    """Demodulate noisy symbols to bits using a slicer.
    
    Parameters
    ----------
    noisy_symbols : np.ndarray
        Array of noisy complex symbols
    slicer : Slicer
        Slicer object containing constellation and decision strategy
        
    Returns
    -------
    np.ndarray
        Array of demodulated bits
    """
    symbol_indices = slicer.slice_symbols(noisy_symbols)
    bits_N = int_to_bits(symbol_indices, slicer.bits_per_symbol)
    return bits_N
