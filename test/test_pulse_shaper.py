from dc_632_26.pulse_shaper import PulseShaper
import dc_632_26.pulses as pul
import numpy as np

def test_pulse_shaper():
    # Setup
    pulse_func = np.arange # generates a linearly increasing pulse
    oversamp = 5
    symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])

    # Test
    pulse_shaper = PulseShaper(pulse_func, oversamp)
    iq = pulse_shaper.generate_waveform(symbols)

    # Validate
    expected_real = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, -1, -2, -3, -4, 0, -1, -2, -3, -4])
    expected_imag = np.array([0, 1, 2, 3, 4, 0, -1, -2, -3, -4, 0, 1, 2, 3, 4, 0, -1, -2, -3, -4])
    np.testing.assert_array_almost_equal(iq.real, expected_real)
    np.testing.assert_array_almost_equal(iq.imag, expected_imag)

# test_pulse_shaper()

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