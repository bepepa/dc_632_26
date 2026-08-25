# This module contains classes and functions for generating and retrieving bit sequences.
"""
This module contains classes and functions for generating and retrieving bit sequences.

It provides both a functional interface and a class-based, object oriented interface for working with bit sequences.

The functional design provides the following functions:
- `random_bit_source(num_bits: int) -> np.ndarray`: Generate a sequence of random bits (0s and 1s) with equal probability.
- `string_source(string: str) -> np.ndarray`: Convert a string into a sequence of bits (0s and 1s), using its byte representation.

Similar to the functional design, the class-based interface provides methods for generating and retrieving bit sequences, encapsulating the state and behavior within objects.
- `RandomBitSource`: Class for generating random bits.
- `TextBitSource`: Class for generating bits from a text string.

The `get_bits(n)` method of the classes is used to retrieve the next `n` bits from the bit source.
"""

import numpy as np


##
# Helper function
##
def byte_to_bits(b: int) -> np.ndarray:
    """convert a single byte to a sequence of 8 bits (MSB first)

    Parameters:
    ----------
    b (int):
        a single byte (0-255)

    Returns:
    --------
    np.ndarray:
        a NumPy vector of bits, stored as uint8

    Example:
    --------
    >>> byte_to_bits(5)
    array([0, 0, 0, 0, 0, 1, 0, 1], dtype=uint8)
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


##
# Functional Design
##
def random_bit_source(num_bits: int) -> np.ndarray:
    """Generate a sequence of random bits (0s and 1s) with equal probability.

    Parameters:
    -----------
    num_bits (int):
        The number of bits to generate.

    Returns:
    --------
    np.ndarray:
        An array of random bits (0s and 1s) of length `num_bits`.

    Example:
    --------
    >>> random_bit_source(5)
    array([0, 1, 0, 1, 1])

    """
    rng = np.random.default_rng()
    return rng.integers(low=0, high=2, size=num_bits, dtype=np.uint8)


def string_source(string: str) -> np.ndarray:
    """convert a string to a vector of bits

    Parameters:
    -----------
    string (str):
        The string to be converted into a bit-sequence; maybe ASCII or UTF-8

    Returns:
    --------
    np.ndarray:
        A numpy array representing the bit sequence of the input string.

    Example:
    --------
    >>> string_source("ABC")
    array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1])
    """
    # convert a string to a sequence of bytes
    bb = string.encode()  # convert the Unicode string to a sequence of bytes
    Nb = len(bb)

    # allocate space
    bits = np.zeros(8 * Nb, dtype=np.uint8)

    for n in range(Nb):
        bits[8 * n : 8 * (n + 1)] = byte_to_bits(bb[n])

    return bits


##
# Class-Based Design
##
class BitSource:
    """Base class for bit sources.

    Methods:
    --------
    get_bits()
        Retrieve the generated bits.
    """

    def __init__(self):
        pass

    def __repr__(self):
        """Return a string representation of the bit source."""
        return f"{self.__class__.__name__}()"

    def get_bits(self, n: int | None) -> np.ndarray:
        """Retrieve the the next n bits.

        Parameters:
        -----------
        n : int, optional
            The number of bits to retrieve. If None, retrieve all available bits.

        Returns:
        --------
        np.ndarray
            Array of retrieved bits.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class RandomBitSource(BitSource):
    """Class for generating random bits.

    Methods:
    --------
    get_bits()
        Retrieve the generated bits.
    """

    def __init__(self):
        super().__init__()

        # Initialize the random number generator
        self.rng = np.random.default_rng()

    def get_bits(self, n: int | None = None) -> np.ndarray:
        """Retrieve the next n random bits.

        Parameters:
        -----------
        n : int, optional
            The number of bits to retrieve. If None, retrieve all available bits.

        Returns:
        --------
        np.ndarray
            Array of retrieved random bits.

        Example:
        --------
        >>> rbs = RandomBitSource()
        >>> rbs.get_bits(5)
        array([0, 1, 0, 1, 1])
        """
        if n is None:
            raise ValueError("Number of bits 'n' must be specified.")

        return self.rng.integers(0, 2, size=n)


class TextBitSource(BitSource):
    """Class for generating bits from a text string.

    Methods:
    --------
    get_bits()
        Retrieve the generated bits.
    """

    def __init__(self, text: str):
        """Initialize the text bit source with the given text.

        Parameters:
        -----------
        text : str
            The text string from which to generate bits.
        """
        super().__init__()
        self.text = text

        # internal state for tracking the bit sequence and current index
        self._bits = self._text_to_bits(text)
        self._index = 0

    def __repr__(self):
        """Return a string representation of the text bit source."""
        return f"{self.__class__.__name__}(text={self.text!r})"

    def _text_to_bits(self, text: str) -> np.ndarray:
        # private helper method to convert text to bits
        bb = text.encode()  # convert the Unicode string to a sequence of bytes
        Nb = len(bb)

        # allocate space
        bits = np.zeros(8 * Nb, dtype=np.uint8)

        for n in range(Nb):
            bits[8 * n : 8 * (n + 1)] = byte_to_bits(bb[n])

        return bits

    def get_bits(self, n: int | None = None) -> np.ndarray:
        """Retrieve the next n bits from the text bit sequence.

        Parameters:
        -----------
        n : int, optional
            The number of bits to retrieve. If None, retrieve all remaining bits.

        Returns:
        --------
        np.ndarray
            Array of retrieved bits.

        Note:
        -----
        If the requested number of bits exceeds the remaining bits, only the available bits will be returned. When an empty array is returned, the source is exhausted and will not provide any more bits.

        Example:
        --------
        >>> tbs = TextBitSource("abc")
        >>> for _ in range(4):
        ...     print(tbs.get_bits(8))
        [0 1 1 0 0 0 0 1]
        [0 1 1 0 0 0 1 0]
        [0 1 1 0 0 0 1 1]
        []
        """
        if n is None:
            # if n is None, retrieve all remaining bits
            n = len(self._bits) - self._index

        if self._index + n > len(self._bits):
            # if the requested number of bits exceeds the remaining bits, adjust n
            # and return only the available bits
            n = len(self._bits) - self._index

        result = self._bits[self._index : self._index + n]
        self._index += n

        return result
