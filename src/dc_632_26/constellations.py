from typing import TypeAlias

import numpy as np
import numpy.typing as npt

NumpyInt: TypeAlias = np.int16 | np.int32 | np.int64


class Constellation:
    def __init__(self, mod_table: npt.NDArray[np.complex128]):
        self.mod_table: npt.NDArray[np.complex128] = np.asarray(
            mod_table, dtype=np.complex128
        )

    @property
    def symbol_energy(self) -> float:
        return float(np.mean(np.abs(self.mod_table) ** 2))

    @property
    def bits_per_symbol(self) -> int:
        return int(np.round(np.log2(len(self.mod_table))))

    @property
    def bit_energy(self) -> float:
        return self.symbol_energy / self.bits_per_symbol

    @property
    def min_distance(self) -> float:
        """
        FIX ME: Summaryline.

        Using Broadcasting feature in numpy arrays
        https://numpy.org/doc/stable/user/basics.broadcasting.html

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
        """
        diff = np.abs(self.mod_table[:, np.newaxis] - self.mod_table[np.newaxis, :])
        np.fill_diagonal(diff, np.inf)  # fill out the 0's
        return float(diff.min())

    @property
    def energy_efficiency(self) -> float:
        return self.min_distance ** 2 / self.bit_energy

    # -- Notations/Aliases that are meaningful in the context of ECE 632 ---
    @property
    def Es(self) -> float:
        return self.symbol_energy

    @property
    def Eb(self) -> float:
        return self.bit_energy

    @property
    def bps(self) -> int:
        return self.bits_per_symbol

    @property
    def dmin(self) -> int:
        return self.min_distance

    @property
    def eta(self) -> int:
        return self.energy_efficiency
    
    def __getitem__(self, bit_seq: int | NumpyInt) -> complex:
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
