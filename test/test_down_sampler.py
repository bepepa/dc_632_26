import pytest
import numpy as np
from dc_632_26.down_sampler import Downsampler
import dc_632_26.pulses as pulses

def test_downsample():
    # Setup
    pulse = pulses.RectangularPulse(8)
    sampling_phase = 2

    downsampler = Downsampler(pulse, sampling_phase)

    x = np.arange(24)

    # Testing
    result = downsampler(x)

    # Expected samples:
    # Start at index 2 and take every 8th sample
    expected = np.array([2, 10, 18])

    # Test
    np.testing.assert_array_equal(result, expected)


def test_uses_pulse_oversamp():
    # Setup
    pulse = pulses.RectangularPulse(5)
    sampling_phase = 1

    downsampler = Downsampler(pulse, sampling_phase)

    x = np.arange(16)

    # Testing
    result = downsampler(x)

    # pulse.oversamp should be 5
    expected = np.array([1, 6, 11])

    # Test
    assert downsampler.oversamp == pulse.oversamp
    np.testing.assert_array_equal(result, expected)


def test_sampling_phase():
    # Setup
    pulse = pulses.RectangularPulse(8)
    x = np.arange(24)

    # Test several possible sampling phases
    for sampling_phase in range(8):
        downsampler = Downsampler(pulse, sampling_phase)

        result = downsampler(x)
        expected = x[sampling_phase::8]

        np.testing.assert_array_equal(result, expected)


def test_downsampler_call():
    # Setup
    pulse = pulses.RectangularPulse(4)
    sampling_phase = 1

    downsampler = Downsampler(pulse, sampling_phase)

    x = np.arange(12)

    # Testing
    result_method = downsampler.downsample(x)
    result_call = downsampler(x)

    # Test that __call__ gives the same result as downsample()
    np.testing.assert_array_equal(result_method, result_call)


def test_invalid_oversamp():
    # Setup
    pulse = pulses.RectangularPulse(0)

    # Test
    with pytest.raises(ValueError):
        Downsampler(pulse, 0)


def test_invalid_sampling_phase():
    # Setup
    pulse = pulses.RectangularPulse(8)

    # Sampling phase cannot be negative
    with pytest.raises(ValueError):
        Downsampler(pulse, -1)

    # Sampling phase must be less than oversamp
    with pytest.raises(ValueError):
        Downsampler(pulse, 8)


def test_rectangular_pulse():
    # Setup
    pulse = pulses.RectangularPulse(8)
    sampling_phase = 3

    downsampler = Downsampler(pulse, sampling_phase)

    x = np.arange(32)

    # Testing
    result = downsampler(x)

    # Test
    expected = x[3::pulse.oversamp]
    np.testing.assert_array_equal(result, expected)


def test_triangle_pulse():
    # Setup
    pulse = pulses.TrianglePulse(8)
    sampling_phase = 2

    downsampler = Downsampler(pulse, sampling_phase)

    x = np.arange(32)

    # Testing
    result = downsampler(x)

    # Test
    expected = x[2::pulse.oversamp]
    np.testing.assert_array_equal(result, expected)


def test_cosine_squared_pulse():
    # Setup
    pulse = pulses.CosineSquaredPulse(8)
    sampling_phase = 4

    downsampler = Downsampler(pulse, sampling_phase)

    x = np.arange(32)

    # Testing
    result = downsampler(x)

    # Test
    expected = x[4::pulse.oversamp]
    np.testing.assert_array_equal(result, expected)


def test_half_sine_pulse():
    # Setup
    pulse = pulses.HalfSinePulse(8)
    sampling_phase = 1

    downsampler = Downsampler(pulse, sampling_phase)

    x = np.arange(32)

    # Testing
    result = downsampler(x)

    # Test
    expected = x[1::pulse.oversamp]
    np.testing.assert_array_equal(result, expected)


def test_different_pulse_shapes():
    # Setup
    pulse_list = [
        pulses.RectangularPulse(8),
        pulses.TrianglePulse(8),
        pulses.CosineSquaredPulse(8),
        pulses.HalfSinePulse(8)
    ]

    sampling_phase = 2
    x = np.arange(32)

    # Testing
    for pulse in pulse_list:
        downsampler = Downsampler(pulse, sampling_phase)

        result = downsampler(x)
        expected = x[sampling_phase::pulse.oversamp]

        # Test
        np.testing.assert_array_equal(result, expected)