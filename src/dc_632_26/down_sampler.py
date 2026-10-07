 import numpy as np 
from dc_632_26.pulses import Pulse


class Downsampler:
    """
    Downsample the matched filter output to the symbol rate.
	
	Parameters for Downsampler:
	-=====================================================================-
    pulse: Pulse
        I used the oversampling factor from 'pulses.py' ('pulse.oversamp').
    start_index: int
        Index of the first sample kept (sampling phase), e.g. the matched
        filter's `delay` plus any desired offset
    oversamp: int, optional
        Overrides the decimation factor. Useful while 'Pulse.oversamp' is
        still tied to `num_samples`, or if samples per symbol differs from
        the pulse length
    sampling_phase: int
        Sampling phase found from the preamble correlation.
        This should be i_hat from the sampling-phase
	-=====================================================================-
    """

    def __init__(self, pulse: Pulse, sampling_phase: int):
        self.oversamp = int(pulse.oversamp)

    def downsample(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)[self.sampling_phase::self.oversamp]

    __call__ = downsample