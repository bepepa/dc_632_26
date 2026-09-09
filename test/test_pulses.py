import dc_632_26.pulses as pulses
import numpy as np

def test_unit_energy():
    # Setup
    num_samples_list = [3, 5, 13, 27, 100]
    
    # Testing
    for num_samples in num_samples_list:
    
        rectangular_pulse = pulses.RectangularPulse(num_samples)
        triangular_pulse = pulses.TrianglePulse(num_samples)
        cosine_squared_pulse = pulses.CosineSquaredPulse(num_samples)
        half_sine_pulse = pulses.HalfSinePulse(num_samples)

        # Test pulse length
        assert len(rectangular_pulse.samples) == num_samples
        assert len(triangular_pulse.samples) == num_samples
        assert len(cosine_squared_pulse.samples) == num_samples
        assert len(half_sine_pulse.samples) == num_samples

        # Test unit energy
        np.testing.assert_almost_equal(np.sum(rectangular_pulse.samples ** 2), 1, 8)
        np.testing.assert_almost_equal(np.sum(triangular_pulse.samples ** 2), 1, 8)
        np.testing.assert_almost_equal(np.sum(cosine_squared_pulse.samples ** 2), 1, 8)
        np.testing.assert_almost_equal(np.sum(half_sine_pulse.samples ** 2), 1, 8)