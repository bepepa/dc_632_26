import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from abc import ABC, abstractmethod


class Pulse(ABC):
    """Base class representing a pulse
    Parameters
    ----------
    oversamp : int
        Oversampling rate / number of samples in one pulse
    nfft : int, optional
        Number of FFT points to use when calculating frequency response. Default is 4096.
    """

    def __init__(self, num_samples, nfft=4096):
        self.num_samples = num_samples
        self.oversamp = num_samples  ## FIX ME: needs to be separate from num_samples
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

    def freq_response(self) -> np.ndarray:
        """Calculate the frequency response of the pulse via Discrete Fourier Transform

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
        freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, d=1 / fs))

        # Plot magnitude
        plt.plot(freqs, np.abs(H), label="DFT")
        try:  # Plot the analytic response, if available
            H_analytic_freqs, H_analytic = self.analytic_freq_response()
            plt.plot(H_analytic_freqs, np.abs(H_analytic), label="Analytic")
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
        """
        freqs = np.linspace(
            -self.oversamp / (2 * T), self.oversamp / (2 * T), self.nfft
        )
        H = np.sqrt(T) * np.sinc(freqs * T)
        return freqs, H


class HalfSinePulse(Pulse):
    """Half-sine pulse

    Parameters
        ----------
        num_samples : int
            Number of samples in the pulse
    """

    def __init__(self, num_samples):
        super().__init__(num_samples)

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
    """

    def __init__(self, num_samples):
        super().__init__(num_samples)

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
    """

    def __init__(self, num_samples):
        super().__init__(num_samples)

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
