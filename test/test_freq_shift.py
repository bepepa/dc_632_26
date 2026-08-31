import pytest
import numpy as np
from dc_632_26.freq_shift import Upconverter


def test_upconverter_process():
    ## parameters
    fc = 10e3
    fs = 40e3  # with these values, carrier cycles through 1, j, -1, -j
    gain = "auto"

    # complex input samples
    input_samples = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)

    upconv = Upconverter(fc=fc, fs=fs, gain=gain)
    output_samples = upconv.process(input_samples)

    assert output_samples.shape == input_samples.shape
    assert np.all(np.isreal(output_samples))
    assert np.allclose(output_samples, np.array([1, -2, -3]) / (3 * np.sqrt(2)))

    # test with a numerical gain
    gain = 2.0
    upconv = Upconverter(fc=fc, fs=fs, gain=gain)
    output_samples = upconv.process(input_samples)

    assert output_samples.shape == input_samples.shape
    assert np.all(np.isreal(output_samples))
    assert np.allclose(output_samples, np.array([2, -4, -6]))
