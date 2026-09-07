import numpy as np


class PulseShaper:
    """Class that generates a sampled waveform given pulse shape and input symbols.

<<<<<<< HEAD
        Parameters
        ----------
        pulse : np.ndarray 
            Discrete-time pulse. Length is equal to the oversampling factor.
        oversamp : int 
            Sampling frequency divided by symbol rate = samples per symbol
    """ 
    def __init__(self, pulse: np.ndarray, oversamp: int):
        self.pulse = pulse
=======
    Parameters
    ----------
    pulse : function
        Function that takes as an argument the number of points to return.
    oversamp : float
        Number of samples per symbol (sample rate divided by symbol rate)
    """

    def __init__(self, pulse, oversamp: float) -> np.ndarray:
        self.pulse = pulse(oversamp)
>>>>>>> 40a6523b5854c3f4eb1a94de4d3a131d9f8a95a5
        self.oversamp = oversamp

    def generate_waveform(self, symbols: np.ndarray) -> np.ndarray:
        """Generates a waveform from the given symbols.

        Parameters:
        -----------
        symbols : np.ndarray
            Array of complex symbols

        Returns:
        --------
        np.ndarray
            Timeseries of the sampled waveform.

        Example:
        >>> pulse_shaper = PulseShaper(pulse, oversamp)
        >>> waveform = pulse_shaper.generate_waveform(symbols)
        """
        # Space the symbols out by the oversamp factor
<<<<<<< HEAD
        waveform_len = (len(symbols)-1)*self.oversamp + 1
        spaced_symbols = np.zeros(waveform_len, dtype=complex) 
        spaced_symbols[::self.oversamp] = symbols
=======
        waveform_len = len(symbols) * self.oversamp
        spaced_symbols = np.zeros(waveform_len, dtype=complex)
        spaced_symbols[:: self.oversamp] = symbols
>>>>>>> 40a6523b5854c3f4eb1a94de4d3a131d9f8a95a5

        # Convolve with pulse to get waveform
        waveform = np.convolve(spaced_symbols, self.pulse)
        return waveform
