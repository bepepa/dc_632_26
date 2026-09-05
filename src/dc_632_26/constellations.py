"""Digital modulation constellations and performance metrics.

This module defines the :class:`Constellation` class, used to represent a digital modulation constellation (e.g., BPSK, QPSK, 8-PSK, 16-QAM) and to compute metrics such as symbol energy, bit energy, minimum distance, and energy efficiency. Several common (gray-encoded) constellations are also instantiated here for convenience.
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

NumpyInt: TypeAlias = np.int16 | np.int32 | np.int64


class Constellation:
    """Represent a digital modulation constellation.

    Parameters
    ----------
    mod_table : numpy.ndarray of numpy.complex128
        Array of complex-valued constellation points (modulation symbols),
        indexed by the bit sequence they represent.

    Attributes
    ----------
    mod_table : numpy.ndarray of numpy.complex128
        Array of complex-valued constellation points.

    """

    def __init__(self, mod_table: npt.NDArray[np.complex128]):
        self.mod_table: npt.NDArray[np.complex128] = np.asarray(
            mod_table, dtype=np.complex128
        )

    @property
    def symbol_energy(self) -> float:
        """Compute the average energy per symbol.

        Returns
        -------
        float
            Mean of the squared magnitude of all constellation points.
        """
        return float(np.mean(np.abs(self.mod_table) ** 2))

    @property
    def bits_per_symbol(self) -> int:
        """Compute the number of bits per symbol.

        Returns
        -------
        int
            Base-2 log of the number of constellation points, rounded to the nearest integer.
        """
        return int(np.round(np.log2(len(self.mod_table))))

    @property
    def bit_energy(self) -> float:
        """Compute average energy per bit.

        Returns
        -------
        float
            Symbol energy divided by the number of bits per symbol
        """
        return self.symbol_energy / self.bits_per_symbol

    @property
    def min_distance(self) -> float:
        """Compute the minimum distance between constellation points.

        Use broadcasting to compute the pairwise distance between every pair of constellation points, then finds the smallest distance.

        Using Broadcasting feature in numpy arrays
        https://numpy.org/doc/stable/user/basics.broadcasting.html

        An example that applies broadcasting to finding QPSK dmin:
            x = QPSK[:, np.newaxis]
            array([[ 1.+1.j],
                [-1.+1.j],
                [-1.-1.j],
                [ 1.-1.j]])

            y = QPSK[np.newaxis, :]
            array([[ 1.+1.j, -1.+1.j, -1.-1.j,  1.-1.j]])

            x-y
            array([[ 0.+0.j,  2.+0.j,  2.+2.j,  0.+2.j],
                [-2.+0.j,  0.+0.j,  0.+2.j, -2.+2.j],
                [-2.-2.j,  0.-2.j,  0.+0.j, -2.+0.j],
                [ 0.-2.j,  2.-2.j,  2.+0.j,  0.+0.j]])

        Returns
        -------
        float
            Smallest distance between any two different constellation points
        """
        diff = np.abs(self.mod_table[:, np.newaxis] - self.mod_table[np.newaxis, :])
        np.fill_diagonal(diff, np.inf)  # fill out the 0's
        return float(diff.min())

    @property
    def energy_efficiency(self) -> float:
        """Compute the energy efficiency of the constellations.

        Returns
        -------
        float
            Squared minimum distance divided by the energy per bit
        """
        return self.min_distance**2 / self.bit_energy

    # -- Notations/Aliases that are meaningful in the context of ECE 632 ---
    @property
    def Es(self) -> float:
        """Return the average symbol energy (alias of `symbol_energy`).

        Returns
        -------
        float
            Average energy per symbol
        """
        return self.symbol_energy

    @property
    def Eb(self) -> float:
        """Return the average bit energy (alias of `bit_energy`).

        Returns
        -------
        float
            Average energy per bit
        """
        return self.bit_energy

    @property
    def bps(self) -> int:
        """Return the number of bits per symbol (alias of `bits_per_symbol`).

        Returns
        -------
        int
            Number of bits per symbol
        """
        return self.bits_per_symbol

    @property
    def dmin(self) -> int:
        """_Return the minimum distance between two different constellation points (alias of `min_distance`).

        Returns
        -------
        int:
            Minimum distance between constellation points
        """
        return self.min_distance

    @property
    def eta(self) -> int:
        """_Return the energy efficiency of the constellation (alias of `energy_efficiency`).

        Returns
        -------
        float:
            Energy efficiency of the constellation.
        """
        return self.energy_efficiency

    def __getitem__(self, bit_seq: int | NumpyInt) -> complex:
        """Look up the constellation point for a given bit sequence.

        Parameters
        ----------
        bit_seq : (int | NumpyInt)
            Index of the constellation points to retrieve

        Returns
        -------
        complex
            Comlex-valued constellation point at `bit_seq`

        """
        return complex(self.mod_table[bit_seq])


bpsk_table: npt.NDArray[np.complex128] = np.array(
    [
        1 + 0j,  # 0
        -1 + 0j,  # 1
    ],
    dtype=np.complex128,
)

qpsk_table = np.array(
    [
        (1 + 1j),  # 0
        (-1 + 1j),  # 1
        (-1 - 1j),  # 2
        (1 - 1j),  # 3
    ],
    dtype=np.complex128,
)

psk8_table = np.array(
    [
        (1 + 0j),  # 0
        np.sqrt(2) / 2 * (1 + 1j),  # 1
        (0 + 1j),  # 2
        np.sqrt(2) / 2 * (-1 + 1j),  # 3
        (-1 + 0j),  # 4
        np.sqrt(2) / 2 * (-1 - 1j),  # 5
        (0 - 1j),  # 6
        np.sqrt(2) / 2 * (1 - 1j),  # 7
    ],
    dtype=np.complex128,
)

qam16_table = np.array(
    [
        (-3 - 3j),  # 0
        (-3 - 1j),  # 1
        (-3 + 3j),  # 2
        (-3 + 1j),  # 3
        (-1 - 3j),  # 4
        (-1 - 1j),  # 5
        (-1 + 3j),  # 6
        (-1 + 1j),  # 7
        (3 - 3j),  # 8
        (3 - 1j),  # 9
        (3 + 3j),  # 10
        (3 + 1j),  # 11
        (1 - 3j),  # 12
        (1 - 1j),  # 13
        (1 + 3j),  # 14
        (1 + 1j),  # 15
    ],
    dtype=np.complex128,
)

BPSK = Constellation(bpsk_table)
QPSK = Constellation(qpsk_table)
PSK8 = Constellation(psk8_table)
QAM16 = Constellation(qam16_table)
