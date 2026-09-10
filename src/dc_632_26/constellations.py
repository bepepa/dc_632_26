"""Digital modulation constellations and helper attributes.

This file defines the :class:`Constellation` class and defines 4 standard constellation objects using this class. 
The constellation class wraps a constellation array as an object for attribute function calls. 

The class accepts either:
  - A numpy array directly (already indexed by bit pattern: 0, 1, 2, ...)
  - A dict with binary literal keys (e.g., {0b00: 1+1j, 0b01: -1+1j, ...})

The resulting object stores the constellation as a numpy array accessible via `mod_table` 
and provides pre-computed helper attributes: symbol energy (Es), bit energy (Eb), 
bits per symbol (bps), minimum distance (dmin), and energy efficiency (eta).

Several common gray-encoded constellations (BPSK, QPSK, 8-PSK, 16-QAM) are pre-instantiated 
for convenience and can be used as: 

standard_constellation.BPSK
standard_constellation.QPSK
standard_constellation.PSK8
standard_constellation.QAM16

"""

from types import SimpleNamespace
import numpy as np
# from helper import symbol_distance

# ============================================================================
# Helper Functions - TODO: Once integration is done, move to a helper.py
# ============================================================================

def symbol_distance(a, b):
    """Compute distance between constellation symbols (works for scalars or arrays)."""
    return np.abs(a - b)

# ============================================================================
# Standard Grey Coded Constellation Tables
# ============================================================================

BPSK_MAP = {
    0b0: 1+0j,
    0b1: -1+0j
}

QPSK_MAP = {
    0b00: 1+1j,
    0b01: -1+1j,
    0b11: -1-1j,
    0b10: 1-1j,
}

PSK8_MAP = {
    0b000: 1+0j,
    0b001: np.sqrt(2)/2 * (1+1j),
    0b011: 0+1j,                   
    0b010: np.sqrt(2)/2 * (-1+1j),  
    0b110: -1+0j,                  
    0b111: np.sqrt(2)/2 * (-1-1j),  
    0b101: 0-1j,                   
    0b100: np.sqrt(2)/2 * (1-1j), 
}

QAM16_MAP = {
    0b0000: -3-3j, 0b0001: -3-1j, 0b0010: -3+3j, 0b0011: -3+1j,
    0b0100: -1-3j, 0b0101: -1-1j, 0b0110: -1+3j, 0b0111: -1+1j,
    0b1000:  3-3j, 0b1001:  3-1j, 0b1010:  3+3j, 0b1011:  3+1j,
    0b1100:  1-3j, 0b1101:  1-1j, 0b1110:  1+3j, 0b1111:  1+1j
}

class Constellation:    
    """Represent a digital modulation constellation.

    Parameters
    ----------
    mod_input : np.ndarray or dict
        Either a numpy array of constellation points indexed by bit sequence,
        or a dict mapping binary literals to symbols: {0b00: 1+1j, 0b01: -1+1j, ...}

    dtype : numpy dtype, optional
        Data type for constellation array (default: np.complex128)

    normalize : bool, optional
        If True, normalize constellation to unit average power (default: False)

    Attributes
    ----------
    mod_table : np.ndarray
        Array of constellation points indexed by bit sequence integer value.

    Es | symbol_energy : float
        Average symbol energy (mean squared magnitude). 

    bps | bits_per_symbol : int
        Number of bits per symbol (log2 of constellation size). 

    Eb | bit_energy : float
        Average energy per bit (symbol_energy / bits_per_symbol). 

    dmin | min_distance : float
        Minimum Euclidean distance between constellation points. 

    eta | energy_efficiency : float
        Energy efficiency (min_distance² / bit_energy).
    """

    def __init__(self, mod_input: np.ndarray | dict, dtype=np.complex128, normalize=False):
        self.dtype = dtype
        
        if isinstance(mod_input, dict):
            self.mod_table = self._dict_to_array(mod_input)
        else:
            self.mod_table = np.asarray(mod_input, dtype=self.dtype)
        
        if normalize:
            avg_power = np.mean(np.abs(self.mod_table)**2)
            self.mod_table = self.mod_table / np.sqrt(avg_power)
            
        self.symbol_energy     = self.Es   = float(np.mean(np.abs(self.mod_table)**2))
        self.bits_per_symbol   = self.bps  = int(np.round(np.log2(len(self.mod_table))))
        self.bit_energy        = self.Eb   = self.symbol_energy / self.bits_per_symbol
        self.min_distance      = self.dmin = self._compute_min_distance()
        self.energy_efficiency = self.eta  = self.min_distance**2 / self.bit_energy

    def _dict_to_array(self, bit_map: dict) -> np.ndarray:
        """Convert binary literal dict to array."""
        max_idx = max(bit_map.keys())
        constellation = np.zeros(max_idx + 1, dtype=self.dtype)
        for idx, sym in bit_map.items():
            constellation[idx] = sym
        return constellation

    def _compute_min_distance(self) -> float:
        """Compute minimum distance between constellation points.
        Uses broadcasting to create NxN matrix of all pairwise distances. 
        """
        diff = symbol_distance(self.mod_table[:, np.newaxis], self.mod_table[np.newaxis, :])  # compute all pairwise distances
        np.fill_diagonal(diff, np.inf)  # Remove self-pair distance, which is 0
        return float(diff.min())  # return min dist

    def point_distance(self, bit_pattern1: int, bit_pattern2: int) -> float:
        """Compute distance between two constellation points based on the constellation map. 
        
        Parameters: bit_pattern1, bit_pattern2 - binary literals (0b00, 0b01) or integer indices
        """
        return float(symbol_distance(self.mod_table[bit_pattern1], self.mod_table[bit_pattern2]))

    def __getitem__(self, bit_pattern: int) -> complex:
        """Return constellation point for bit sequence index."""
        return complex(self.mod_table[bit_pattern])

    def __call__(self):
        """Make Constellations object callable. Returns Mod table as numpy array"""
        return self.mod_table

standard_constellation = SimpleNamespace(
    BPSK=Constellation(BPSK_MAP,normalize=True),
    QPSK=Constellation(QPSK_MAP,normalize=True),
    PSK8=Constellation(PSK8_MAP,normalize=True),
    QAM16=Constellation(QAM16_MAP,normalize=True)
)
# Allow legacy access
BPSK=Constellation(BPSK_MAP,normalize=False)
QPSK=Constellation(QPSK_MAP,normalize=False)
PSK8=Constellation(PSK8_MAP,normalize=False)
QAM16=Constellation(QAM16_MAP,normalize=False)