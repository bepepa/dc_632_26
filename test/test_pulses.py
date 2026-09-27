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

def test_triangle_pulse():
    """
    Unit test to confirm:
    - Using a non-default Symbol period T
    - Proper FT returned. Testing correct scaling at H(0)
    - Correct lengths: len(freqs) = len(H)
    - Zero-corssing of sinc function
    - Check against numerical DFT
    """
    num_samples = 201

    # Non-default symbol period
    T = 2.0

    triangle_pulse = pulses.TrianglePulse(num_samples)
    freqs, H = triangle_pulse.analytic_freq_response(T)

    # Check for H(0) == sqrt(3 * T) / 2 @ T = 2.0
    dc_index = np.argmin(np.abs(freqs))
    assert H[dc_index] == pytest.approx(np.sqrt(3*T)/2, rel=1e-3)

    # Check correct lengths
    assert len(freqs) == len(H)

    # First zero of the the sinc^2(f * T / 2) is at f = 2 / T or sinc(1)
    zero_index = np.argmin(np.abs(freqs - 2 / T))
    assert H[zero_index] == pytest.approx(0, abs=1e-3)

    # Checking against numerical DFT
    """
    Note:

    For this test, the T of the test DFT must be equal to the oversamp
    value, which is locked into the number of samples (with a FIX ME).
    No alternate value of T can result in a match because the it is
    testing against a generated pulse.
    """

    T_DFT = float(triangle_pulse.oversamp)
    freqs_T_DFT, H_T_DFT = triangle_pulse.analytic_freq_response(T_DFT)
    dc_index_dft = np.argmin(np.abs(freqs_T_DFT))

    H_DFT = np.fft.fftshift(triangle_pulse.freq_response())
    dft_freqs = np.fft.fftshift(
        np.fft.fftfreq(triangle_pulse.nfft, d=T_DFT / triangle_pulse.oversamp)
    )
    dft_dc_index = np.argmin(np.abs(dft_freqs))
    assert np.abs(H_DFT[dft_dc_index]) == pytest.approx(H_T_DFT[dc_index_dft], rel=0.05)



