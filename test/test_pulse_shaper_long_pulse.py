from dc_632_26.pulse_shaper import PulseShaper
from dc_632_26.pulses import half_sine
from dc_632_26.pulses import cosine_squared
import numpy as np
import matplotlib.pyplot as plt



def test_pulse_shaper_long_pulse():
    # Setup
    # pulse_func = np.arange # generates a linearly increasing pulse
    #pulse_func=half_sine
    #oversamp = 5
    #symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])
    #pulse_func = lambda N: np.arange(5)   # test to make pulse longer than waveform, code doesn't break 
   
   # pulse_func = lambda N: np.ones(5) # Expected IQ plots
    pulse_func = lambda N: np.ones(100) # Setting pulse length longer than wavelength length try ones(100) 
                                        # does not produce desirable IQ
    oversamp = 20
    symbols = np.array([1+1j, 1-1j])

    # Test
    pulse_shaper = PulseShaper(pulse_func, oversamp)
    iq = pulse_shaper.generate_waveform(symbols)

    # Validate
    # expected_real = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, -1, -2, -3, -4, 0, -1, -2, -3, -4])
    # expected_imag = np.array([0, 1, 2, 3, 4, 0, -1, -2, -3, -4, 0, 1, 2, 3, 4, 0, -1, -2, -3, -4])
    # np.testing.assert_array_almost_equal(iq.real, expected_real)
    # np.testing.assert_array_almost_equal(iq.imag, expected_imag)

    print(f"pulse length",len(pulse_shaper.pulse),"waveform length",len(iq))
    print("pulse length:", len(pulse_shaper.pulse))
    print("waveform length:", len(iq))
    print("first few samples:", iq[:10])

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(iq.real)
    plt.ylabel("I real")
    plt.title("Waveform")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(iq.imag)
    plt.ylabel("Q imag")
    plt.title("Waveform")
    plt.grid(True)
    plt.show()  

test_pulse_shaper_long_pulse()
# if __name__ == "__main__":
#     test_pulse_shaper()