from typing import TypeAlias

import numpy as np
import numpy.typing as npt

NumpyInt: TypeAlias = np.int16 | np.int32 | np.int64


class Constellation:
    def __init__(self, mod_table: npt.NDArray[np.complex128]):
        self.mod_table: npt.NDArray[np.complex128] = np.asarray(
            mod_table, dtype=np.complex128
        )

    def get_symbol_energy(self) -> np.float64:
        return np.mean(np.abs(self.mod_table) ** 2)

    def get_number_bits_per_symbol(self) -> np.uint64:
        return np.uint64(np.round(np.log2(len(self.mod_table))))

    def get_bit_energy(self) -> np.float64:
        return self.get_symbol_energy() / self.get_number_bits_per_symbol()

    def get_min_distance(self) -> np.float64:
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
        return diff.min()

    def get_energy_efficiency(self) -> np.float64:
        return self.get_min_distance() ** 2 / self.get_bit_energy()

    def __getitem__(self, bit_seq: int | NumpyInt) -> np.complex128:
        return self.mod_table[bit_seq]


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
