import numpy as np


class PulseShaper:
    """Class that generates a sampled waveform given pulse shape, oversampling factor, and input symbols.

        Parameters
        ----------
        pulse : Callable
            A Pulse instance (callable) that returns pulse samples. Plain functions
            are not supported unless adapted.
        oversamp : int
            Number of samples per symbol (sample rate divided by symbol rate)
            Provided to the Pulse instance so it generates the correct number of
            samples per symbol.
    """ 
    def __init__(self, pulse):

        self.pulse = pulse
        self.oversamp = pulse.oversamp

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

        # Get pulse samples 
        pulse_samples = self.pulse()
        # Convolve with pulse to get waveform
        waveform = np.convolve(spaced_symbols, pulse_samples)
        #waveform = waveform[:waveform_len] #clipping, try removing clipping here 
        return waveform
        