import numpy as np


class PulseShaper:
    """Class that generates a sampled waveform given pulse shape and input symbols.

        Parameters
        ----------
        pulse : np.ndarray 
            Discrete-time pulse. Length is equal to the oversampling factor.
    """ 
    def __init__(self, pulse: np.ndarray):
        self.pulse = pulse
        self.oversamp = len(pulse)

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
        waveform_len = len(symbols)*self.oversamp
        spaced_symbols = np.zeros(waveform_len, dtype=complex) 
        spaced_symbols[::self.oversamp] = symbols

        # Convolve with pulse to get waveform
        waveform = np.convolve(spaced_symbols, self.pulse)
        waveform = waveform[:waveform_len]
        return waveform
