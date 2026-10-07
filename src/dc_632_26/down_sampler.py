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

        if self.oversamp < 1:
            raise ValueError("pulse.oversamp must be a positive integer")

        if int(sampling_phase) != sampling_phase or sampling_phase < 0:
            raise ValueError(
                "sampling_phase must be a non-negative integer"
            )

        if sampling_phase >= self.oversamp:
            raise ValueError(
                "sampling_phase must be less than pulse.oversamp"
            )

        self.sampling_phase = int(sampling_phase)

    def downsample(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)[self.sampling_phase::self.oversamp]

    __call__ = downsample