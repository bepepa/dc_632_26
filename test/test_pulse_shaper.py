from dc_632_26.pulse_shaper import PulseShaper
import dc_632_26.pulses as pul
import numpy as np

def test_pulse_shaper():
    """Nominal test case with full response pulse.
    """
    # Setup
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
    """Partial response pulse (pulse length is greater than oversampling factor)
    """
    # Setup
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

def test_pulses():
    
    # Setup
    num_samples_list = [3, 5, 13, 27, 100]
    
    # Testing
    for num_samples in num_samples_list:
    
        rectangular_pulse = pul.rectangular(num_samples)
        triangular_pulse = pul.triangular(num_samples)
        cosine_squared_pulse = pul.cosine_squared(num_samples)
        half_sine_pulse = pul.half_sine(num_samples)
        
        np.testing.assert_almost_equal(np.sum(rectangular_pulse ** 2), 1, 8)
        np.testing.assert_almost_equal(np.sum(triangular_pulse ** 2), 1, 8)
        np.testing.assert_almost_equal(np.sum(cosine_squared_pulse ** 2), 1, 8)
        np.testing.assert_almost_equal(np.sum(half_sine_pulse ** 2), 1, 8)
        
#test_pulses()