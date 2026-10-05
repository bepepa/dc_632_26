"""Matched filter for the pulse shapes defined in `dc_632_26.pulses`.

`MatchFilter.filter` runs at the input's sample rate (no downsampling to the symbol rate).
"""

import numpy as np

from dc_632_26.pulses import Pulse


class MatchFilter:
    """Match filter for a given pulse shape.

    This filter is a time-reversed, and conjugated, version of the input pulse

    Parameters
    ----------
    pulse : Pulse
        Pulse shape to match against.

    Attributes
    ----------
    taps : np.ndarray
        Filter coefficients. `taps` is causal (`taps[n] = 0` for `n < 0`) simply by virtue
        of being an ordinary array. No separate delay stage is needed.

    num_taps : int
        Number of filter taps.

    delay : int
        When convolving a Pulse with its Matched Filter, delay is the index
        where the match-filter produces a peak.
    """

    def __init__(self, pulse: Pulse):
        self.taps = np.conj(pulse.samples[::-1])
        self.num_taps = len(self.taps)
        self.delay = int(self.num_taps - 1)

    def filter(self, x: np.ndarray) -> np.ndarray:
        """_Apply the matched filter to an input signal.

        Use "full" convolution, so the output is longer than the input by
        `num_taps - 1` samples. The output sample rate is the same as the
        input, e.g. still `fsT` samples/symbol.

        Parameters
        ----------
        x : np.ndarray
            Input signal, samples at fsT samples per symbol

        Returns
        -------
        np.ndarray:
            Matched-filter output, same sample rate as `x`, length `len(x) + self.num_taps -1`.
        """
        return np.convolve(x, self.taps, mode="full")
