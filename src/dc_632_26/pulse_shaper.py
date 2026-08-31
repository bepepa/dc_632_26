import numpy as np

class PulseShaper:
    """Class that generates a sampled waveform given pulse shape, oversampling factor, and input symbols.

        Parameters
        ----------
        pulse : function
            Function that takes as an argument the number of points to return.
        oversamp : float
            Number of samples per symbol (sample rate divided by symbol rate)
    """ 
    def __init__(self, pulse:function, oversamp:float) -> np.ndarray:
        self.pulse = pulse(oversamp)
        self.oversamp = oversamp

    def generate_waveform(self, symbols:np.ndarray) -> np.ndarray:
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
        