import numpy as np
import matplotlib.pyplot as plt   
import warnings  
from abc import ABC, abstractmethod

<<<<<<< HEAD
class Pulse(ABC):
    """Base class representing a pulse
        Parameters
        ----------
        oversamp : int
            Oversampling rate / number of samples in one pulse
    """ 
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.samples = self._generate_samples()
        self.nfft = 4096

    @abstractmethod
    def _generate_samples(self):
        """Subclasses must implement.
        Generates the discrete-time pulse samples."""
        pass

    def _normalize_energy(self, samples: np.ndarray) -> np.ndarray:
        """Normalizes samples to unit energy

        Parameters
        ----------
        samples : np.ndarray
            Samples to normalize

        Returns
        -------
        np.ndarray
            Normalized samples
        """
        energy = np.sum(samples**2)
        return samples / np.sqrt(energy)

    def analytic_freq_response(self):
        """Optional method to retun the analytic frequency response of the pulse, starting at freq 0. 
        Implemented by subclass.
        """
        raise NotImplementedError("Not yet implemented by subclass")

    def freq_response(self) -> np.ndarray:
        """Calculate the frequency of the pulse via Discrete Fourier Transform

        Returns
        -------
        np.ndarray
            Discrete Fourier Transform of the pulse
        """
        return np.fft.fft(self.samples, n=self.nfft)
    
    def plot_freq_response(self, fs: float):
        """Creates a plot of the pulse's frequency response (magnitude only)

        Plots the DFT of the pulse, as well as the analytic frequency response if defined.

        Parameters
        ----------
        fs : float
            Sampling frequency
        """
        # Get frequency response
        H = np.fft.fftshift(self.freq_response())
        # Frequency axis in Hz
        freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, d=1/fs))

        #Plot magnitude
        plt.plot(freqs, np.abs(H), label='DFT')
        try: # Plot the analytic response, if available
            H_analytic = self.analytic_freq_response()
            plt.plot(freqs, np.abs(H_analytic), label='Analytic')
            plt.legend()
        except NotImplementedError:
            pass
        plt.title("Pulse Frequency Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("|H(f)|")
        plt.grid(True)
        plt.show()


class RectangularPulse(Pulse):
    """Rectangular pulse
    Parameters
        ----------
        num_samples : int
            Number of samples in the pulse
    """

    def __init__(self, num_samples):
        super().__init__(num_samples)

    def _generate_samples(self):
        samples = np.ones(self.num_samples)
        return self._normalize_energy(samples)


class HalfSinePulse(Pulse):
    """Half-sine pulse

        Parameters
            ----------
            num_samples : int
                Number of samples in the pulse
        """
    
    def __init__(self, num_samples):
        super().__init__(num_samples)
=======
# Defining the rectangular pulse
def rectangular(num_samples):
    
    '''
    Parameters:
    ------------
    num_samples (int): number of samples
    
    Returns:
    ------------
    pulse (np.array): normalized array of samples according to the pulse shape
    '''
    
    # Creating the rectangular shape
    pulse = np.ones(num_samples)
    
    # Normalizing to unit energy
    return pulse / np.linalg.norm(pulse)

# Defining the triangular pulse
def triangular(num_samples):
    
    '''
    Parameters:
    ------------
    num_samples (int): number of samples
    
    Returns:
    ------------
    pulse (np.array): normalized array of samples according to the pulse shape
    '''
    
    # Time axis
    t = np.arange(num_samples) / num_samples
    
    # Creating the triangular shape
    pulse = np.piecewise(t,
                         [((0 <= t) & (t < 0.5)),
                          ((0.5 <= t) & (t < 1))],
                         [lambda t: t,
                          lambda t: 1 - t])
    
    # Normalizing to unit energy
    return pulse / np.linalg.norm(pulse)

# Defining the cosine squared pulse
def cosine_squared(num_samples):
    
    '''
    Parameters:
    ------------
    num_samples (int): number of samples
    
    Returns:
    ------------
    pulse (np.array): normalized array of samples according to the pulse shape
    '''
    
    # Time axis
    t = np.arange(num_samples) / num_samples
    
    # Creating the cosine squared shape
    pulse = 1 / 2 - 1 / 2 * np.cos(2 * np.pi * t)
    
    # Normalizing to unit energy
    return pulse / np.linalg.norm(pulse)
>>>>>>> 5aac737d9a72d6f4787e5f41ef2f1fd29d2533c4

    def _generate_samples(self):
        return self._normalize_energy(np.sin(np.pi * np.linspace(0., 1., self.num_samples, endpoint=False)))

    
class CosineSquaredPulse():
    """Cosine-squared pulse

        Parameters
            ----------
            num_samples : int
    """

    def __init__(self, num_samples):
        super().__init__(num_samples)

    def _generate_samples(self):
        pass # placeholder
    