from dc_632_26.pulse_shaper import PulseShaper
from dc_632_26.pulses import RectangularPulse
from dc_632_26.pulses import HalfSinePulse
import numpy as np
import matplotlib.pyplot as plt

"""Class that supports pulse shaping utilities.

    This module tests two pulse types used for waveform generation,
    including rectangle, and half-sine pulses with a plot

    
    TODO
    ----------
    Add option for checking with PLOT=0 (ie validate iq)
    Test other pulses
    Check long pulse issues 

   
""" 

def test_pulse_shaper():
    # Setup
    #pulse_func = np.arange # generates a linearly increasing pulse
    fs=1 #Hz for testing only
    #pulse = RectangularPulse(num_samples=4, oversamp=20)
    pulse = RectangularPulse(num_samples=4, oversamp=64)
    #pulse = HalfSinePulse(num_samples=20, oversamp=20)
    symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])
    #symbols = (np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=256))

    # Test
    pulse_shaper = PulseShaper(pulse)
    iq = pulse_shaper.generate_waveform(symbols)
    PLOT=1

    # Validate
    # expected_real = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, -1, -2, -3, -4, 0, -1, -2, -3, -4])
    # expected_imag = np.array([0, 1, 2, 3, 4, 0, -1, -2, -3, -4, 0, 1, 2, 3, 4, 0, -1, -2, -3, -4])
    # np.testing.assert_array_almost_equal(iq.real, expected_real)
    # np.testing.assert_array_almost_equal(iq.imag, expected_imag)

    print(f"pulse length",len(pulse_shaper.pulse()),"waveform length",len(iq))
    print("waveform length:", len(iq))
    print("first few samples:", iq[:10])

    if PLOT:
         # --- First figure: waveform ---
        plt.figure(1)
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
        

        # --- First figure: fft ---
        plt.figure(2)
        pulse.plot_freq_response(fs)

        plt.show()
    
   

test_pulse_shaper()