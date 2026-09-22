import dc_632_26.pulses as pulses
import pytest
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

def test_pulse_class():
    '''Confirms that Pulse subclasses must implement a _generate_samples method'''

    # Setup: Create a pulse class without defining _generate_samples
    class TestPulse(pulses.Pulse):
        def __init__(self, num_samples):
                super().__init__(num_samples)

    # Test
    with pytest.raises(TypeError) as excinfo:
        pulse = TestPulse(5)
    assert "Can't instantiate abstract class TestPulse without an implementation for abstract method '_generate_samples'" in str(excinfo.value)