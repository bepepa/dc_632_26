import numpy as np
from dc_632_26.pulses import Pulse


class PulseShaper:
    """Class that generates a sampled waveform given pulse shape and input symbols.

        Parameters
        ----------
        pulse : Pulse
            Pulse object
        oversamp : int 
            Sampling frequency divided by symbol rate = samples per symbol
    """ 
    def __init__(self, pulse: Pulse, oversamp: int):
        self.pulse = pulse
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
        waveform_len = (len(symbols)-1)*self.oversamp + 1
        spaced_symbols = np.zeros(waveform_len, dtype=complex) 
        spaced_symbols[::self.oversamp] = symbols

        # Convolve with pulse to get waveform
        waveform = np.convolve(spaced_symbols, self.pulse.samples)
        return waveform
