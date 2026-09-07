from dc_632_26.pulse_shaper import PulseShaper
import numpy as np

def test_pulse_shaper():
    # Setup: Pulse length = oversamp
    oversamp = 5
    pulse_length = oversamp
    pulse = np.arange(oversamp) # generates a linearly increasing pulse
    symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])

    # Test
    pulse_shaper = PulseShaper(pulse, oversamp)
    iq = pulse_shaper.generate_waveform(symbols)

    # Validate
    expected_real = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, -1, -2, -3, -4, 0, -1, -2, -3, -4])
    expected_imag = np.array([0, 1, 2, 3, 4, 0, -1, -2, -3, -4, 0, 1, 2, 3, 4, 0, -1, -2, -3, -4])
    np.testing.assert_array_almost_equal(iq.real, expected_real)
    np.testing.assert_array_almost_equal(iq.imag, expected_imag)

def test_long_pulse():
    # Setup: Pulse length = oversamp
    oversamp = 3
    pulse_length = 5
    pulse = np.arange(pulse_length) # generates a linearly increasing pulse
    symbols = np.array([1, -1, 2])

    # Test
    pulse_shaper = PulseShaper(pulse, oversamp)
    iq = pulse_shaper.generate_waveform(symbols)

    # Validate
    expected_real = np.array([0, 1, 2, 3, 3, -2, -3, -2, 4, 6, 8])
    expected_imag = np.zeros(11)
    np.testing.assert_array_almost_equal(iq.real, expected_real)
    np.testing.assert_array_almost_equal(iq.imag, expected_imag)