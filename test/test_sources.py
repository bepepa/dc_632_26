# Unit tests for `dc_632_26.sources` module
import pytest
import numpy as np

from dc_632_26.sources import (
    random_bit_source,
    string_source,
    RandomBitSource,
    TextBitSource,
)


def test_random_bit_source():
    """Unit test for the random_bit_source function."""

    num_bits = 10

    bits = random_bit_source(num_bits)
    assert isinstance(bits, np.ndarray), "Output should be a numpy array"
    assert bits.shape[0] == num_bits, "Output array should have the correct length"
    assert np.all(np.isin(bits, [0, 1])), "All elements should be 0 or 1"


def test_string_source():
    """Unit test for the string_source function."""
    text = "A"
    expected_bits = np.array([0, 1, 0, 0, 0, 0, 0, 1], dtype=np.uint8)
    assert np.array_equal(
        string_source(text), expected_bits
    ), "Test failed for input 'A'"

    text = "Äö"
    expected_bits = np.array(
        [
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,  # Ä (U+00C4) in UTF-8: 0xC3 0x84
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            0,  # ö (U+00F6) in UTF-8: 0xC3 0xB6
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(
        string_source(text), expected_bits
    ), "Test failed for input 'Äö'"


def test_RandomBitSource():
    """Unit test for the RandomBitSource class."""
    rbs = RandomBitSource()
    num_bits = 10

    bits = rbs.get_bits(num_bits)
    assert isinstance(bits, np.ndarray), "Output should be a numpy array"
    assert bits.shape[0] == num_bits, "Output array should have the correct length"
    assert np.all(np.isin(bits, [0, 1])), "All elements should be 0 or 1"


def test_TextBitSource():
    """Unit test for the TextBitSource class."""
    tbs = TextBitSource("abc")
    for _ in range(3):
        assert tbs.get_bits(8).shape == (8,)
    # after consuming all bits, the next call should return an empty array
    assert tbs.get_bits(8).shape == (0,)

    # test retrieving all remaining bits
    tbs = TextBitSource("abc")
    all_bits = tbs.get_bits()
    assert all_bits.shape == (24,)
    # after consuming all bits, the next call should return an empty array
    assert tbs.get_bits().shape == (0,)
