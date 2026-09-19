import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from abc import ABC, abstractmethod


class Pulse(ABC):
    """Base class representing a pulse
    Parameters
    ----------
    num_samples : int
        Number of samples in the pulse
    oversamp : int, optional
        Oversampling rate (samples per symbol period). Defaults to num_samples (full-response pulse).
    nfft : int, optional
        Number of FFT points to use when calculating frequency response. Default is 1024.
    """

    def __init__(self, num_samples: int, oversamp: int=None, nfft: int=4096):
        self.num_samples = num_samples
        if oversamp is None:
            self.oversamp = num_samples 
        else:
            self.oversamp = oversamp
        self.samples = self._generate_samples()
        self.nfft = nfft

    @abstractmethod
    def _generate_samples(self) -> np.ndarray:
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

    def analytic_freq_response(self, T: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Optional method to return the analytic frequency response of the pulse.

        Parameters
        ----------
        T : float
            Duration of the pulse in seconds. Default is 1.0.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Frequencies and corresponding frequency response values
            The frequency response is calculated analytically by the subclass, if implemented.
        Notes
        -----
        Implemented by subclass.
        """
        raise NotImplementedError("Not yet implemented by subclass")

    def numerical_freq_response(
        self, 
        T: float = 1.0
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the frequency response of the pulse via Discrete Fourier Transform
                Parameters
                ----------       
                T : float
                    Duration of the pulse in seconds. Default is 1.0 and frequency axis is
                    Normalized frequency(1/T) or Multiples of Symbol rate
            
                    If specific time is passed, the frequency axis is Frequency Hz and spans
                    -fs/2 to fs/2 
        
                Returns
                -------
                Tuple[np.ndarray, np.ndarray]
                    freq_axis : np.ndarray
                        Frequency vector ranging from -fs/2 to fs/2 
                    freq_resp : np.ndarray
                        The centered, complex discrete frequency response (FFT shifted).
                
                Notes
                -----
                Implemented by subclass.
                The frequency response is scaled by np.sqrt(self.num_samples) because 
                the time-domain samples have been normalized to unit energy. This scaling 
                ensures that the numerical DFT magnitude matches the continuous analytic 
                frequency response peak.
                """

        # This line sets a value for nfft if none has been specified.
        N_points = getattr(self, 'nfft', None) or max(len(self.samples), 1024)
        print(f"T=",T)
        #Derive sampling frequency and axis vector
        if T==1:
            fs=self.oversamp
            freq_axis =np.linspace(
             -self.num_samples / (2 * T), self.num_samples / (2 * T), N_points
            )
        else:
            fs = self.oversamp / T
            freq_axis = np.fft.fftshift(np.fft.fftfreq(N_points, d=1/fs))

        # Compute the numerical frequency response
        # The scaling below is because the energy of the samples has been normalized to one, 
        # but as in HW 2, problem 3, part d, the discrete time approximation of the energy is fs. 
        #So to make the DFT, match the analytic freq response, we scale as below.
        freq_resp = np.fft.fftshift(np.fft.fft(self.samples, n=N_points))/ np.sqrt(fs)
    
        return freq_axis, freq_resp
       

    def plot_freq_response(self, fs: float,T: float = 1.0):
        """Creates a plot of the pulse's frequency response (magnitude only)

        Plots the DFT of the pulse, as well as the analytic frequency response if defined.

        Parameters
        ----------
        fs : float
            Sampling frequency
        """
        
        #Get numerical frequency response
        # Note to compare against current analytical freq response, calling with default T=1 
        # and using normalized frequency 1/T x axis
        print(f"T=",T)
        if T==1:
            H_numerical_freqs, H =self.numerical_freq_response()
            plt.plot(H_numerical_freqs, np.abs(H), label="DFT")

            try:  # Plot the analytic response, if available
                H_analytic_freqs, H_analytic = self.analytic_freq_response()
                plt.plot(H_analytic_freqs, np.abs(H_analytic), '--',label="Analytic")
                plt.legend()
            except NotImplementedError:
                pass
            plt.title("Pulse Frequency Response")
            plt.xlabel("Normalized Frequency (1/T)") 
            plt.ylabel("|H(f)|")
            plt.grid(True)
            plt.show()
        else:
            H_numerical_freqs, H =self.numerical_freq_response(T=T)
            plt.plot(H_numerical_freqs, np.abs(H), label="DFT")
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

    def analytic_freq_response(self, T: float = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Optional method to return the analytic frequency response of the pulse.

        Parameters
        ----------
        T : float
            Duration of the pulse in seconds. Default is 1.0.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Frequencies and corresponding frequency response values
            The frequency response is calculated analytically by the subclass, if implemented.
        """
        freqs = np.linspace(
            -self.num_samples / (2 * T), self.num_samples / (2 * T), self.nfft
        )
        H = np.sqrt(T) * np.sinc(freqs * T)
        return freqs, H


class HalfSinePulse(Pulse):
    """Half-sine pulse

    Parameters
    ----------
    num_samples : int
        Number of samples in the pulse
    nfft : int, optional
        Number of FFT points to use when calculating frequency response. Default is 4096.
    """

    def __init__(self, num_samples: int, nfft: int=4096):
        super().__init__(num_samples, nfft=nfft)

    def _generate_samples(self):
        return self._normalize_energy(
            np.sin(np.pi * np.linspace(0.0, 1.0, self.num_samples, endpoint=False))
        )


class CosineSquaredPulse(Pulse):
    """Cosine-squared pulse

    Parameters
    ----------
    num_samples : int
        Number of samples in the pulse
    nfft : int, optional
        Number of FFT points to use when calculating frequency response. Default is 4096.
    """

    def __init__(self, num_samples: int, nfft: int=4096):
        super().__init__(num_samples, nfft=nfft)

    def _generate_samples(self):
        # Time axis
        t = np.arange(self.num_samples) / self.num_samples

        # Creating the cosine squared shape
        pulse = 1 / 2 - 1 / 2 * np.cos(2 * np.pi * t)

        # Normalizing to unit energy
        return self._normalize_energy(pulse)


class TrianglePulse(Pulse):
    """Triangular pulse

    Parameters
    ----------
    num_samples : int
        Number of samples in the pulse
    nfft : int, optional
        Number of FFT points to use when calculating frequency response. Default is 4096.
    """

    def __init__(self, num_samples: int, nfft: int=4096):
        super().__init__(num_samples, nfft=nfft)

    def _generate_samples(self):
        # Time axis
        t = np.arange(self.num_samples) / self.num_samples

        # Creating the triangular shape
        pulse = np.piecewise(
            t,
            [((0 <= t) & (t < 0.5)), ((0.5 <= t) & (t < 1))],
            [lambda t: t, lambda t: 1 - t],
        )

        # Normalizing to unit energy
        return self._normalize_energy(pulse)
