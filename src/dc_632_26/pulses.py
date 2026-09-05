import numpy as np
import matplotlib.pyplot as plt   
import warnings  
  
"""Class that supports pulse shaping utilities.

    This module defines several pulse types used for waveform generation,
    including rectangle, and half-sine pulses and 
    placeholders for additional shapes such
    as cosine squared.

    The pulses are returned as NumPy arrays suitable for use with the
    pulse_shaper module.

    References
    ----------
    copilot - Used for python references with caution and checking 

    https://numpy.org/devdocs/reference/generated/numpy.fft.fft.html

    TODO
    ----------
    pull fs, oversamp, and symbol_rate from Transmitter class once integrated (Parent class)

    normalize_energy

    cosine_squared pulse
""" 

class Pulse:

    def __init__(self, num_samples, oversamp, is_analytic):
        self.num_samples = num_samples
        self.oversamp = oversamp
        self.is_analytic = is_analytic

    def generate(self):
        raise NotImplementedError("Subclasses must implement generate()")

    def __call__(self):
        return self.generate()

    def normalize_energy(self):
        """
        TODO: Compute pulse energy.
    
        """
        pass

    # def freq_response(self):
    #     if self.is_analytic and hasattr(self, "analytic_freq_response"):
    #         return self.analytic_freq_response()
    #     else:
    #         return np.fft.fft(self())

    def freq_response(self):
        if self.is_analytic:
            H = self.analytic_freq_response()
            if H is not None:
                return H
        # use  numeric FFT if analytic_freq_response not available yet
        return np.fft.fft(self.generate(),n=4096)
        
    def plot_freq_response(self,fs):
        # Get frequency response (FFT or analytic)
        H = self.freq_response()
        # Number of points
        N = len(H)
        # Frequency axis in Hz
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
        # Shifted frequency response
        Hs = np.fft.fftshift(H)
        #Plot magnitude
        plt.plot(freqs, np.abs(Hs))
        plt.title("Pulse Frequency Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("|H(f)|")
        plt.grid(True)
        plt.show()


class RectangularPulse(Pulse):

    """
    Parameters
        ----------
        num_samples : int
            Number of samples in the pulse
        oversamp : int
            Number of samples per symbol (sample rate divided by symbol rate)

        Returns
        ---------
        pulse : np.ndarray
            HalfSine pulse of length `num_samples`.

            analytic_freq_response : np.ndarray.
    """

    def __init__(self, num_samples, oversamp):
        super().__init__(num_samples, oversamp, is_analytic=False)

    def generate(self):
        return np.ones(self.num_samples)

class HalfSinePulse(Pulse):
    """
        Parameters
            ----------
            num_samples : int
                Number of samples in the pulse
            oversamp : int
                Number of samples per symbol (sample rate divided by symbol rate)
    
            Returns
            ---------
            pulse : np.ndarray
                HalfSine pulse of length `num_samples`. 

            Notes
            -----
            `is_analytic=True` is set, but the analytic frequency response is
            not yet implemented. A warning is issued and the numeric FFT fallback
            is used instead.

            TODO
            ----
            Implement analytic frequency response once symbolic form is finalized.
        """
    
    def __init__(self, num_samples, oversamp):
        super().__init__(num_samples, oversamp, is_analytic=True)

    def analytic_freq_response(self):
        warnings.warn(
        "Analytic frequency response not yet implemented; using numeric FFT.",
        UserWarning
    )
        return None

    def generate(self):
        #pulse=np.sqrt(2) * np.sin(np.pi * np.linspace(0., 1., self.num_samples, endpoint=False))
        return np.sqrt(2) * np.sin(np.pi * np.linspace(0., 1., self.num_samples, endpoint=False))

    
class CosineSquaredPulse():

    """
    Cosine-squared pulse placeholder.

    TODO: 
        Implement generate() 
        Implement analytic frequency response once symbolic form is finalized.
        
    """

    def __init__(self, num_samples, oversamp):
        super().__init__(num_samples, oversamp, is_analytic=True)

    def generate(self):
        raise NotImplementedError(
            "CosineSquaredPulse.generate() not implemented yet."
        )
    
    def analytic_freq_response(self):
            warnings.warn(
            "Analytic frequency response not yet implemented; using numeric FFT.",
            UserWarning
        )
            return None
    # def cosine_squared(num_samples):
    
    # time_grid = np.arange(num_samples) / num_samples
    # pulse = np.sqrt(8 / 3) * (1 / 2 - 1 / 2 * np.cos(2 * np.pi * time_grid))
    
    # return pulse
    
       