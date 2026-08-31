# freq_shift module for DC 632 26 course
"""
This module provides functionality for frequency shifting in the DC 632 26 package.

It provides the following:
    - Upconverter: A class for frequency shifting (upconversion) of complex baseband signals.
"""

import numpy as np


##
# upconverter functionality
##
class Upconverter:
    """
    Class representing an upconverter device.

    Attributes:
    -----------
        fc: The carrier frequency of the upconverter.
        fs: The sample rate of the upconverter.
        gain: The gain of the upconverter.

    Methods:
    --------
        process(input_samples: np.ndarray) -> np.ndarray
            Process the complex baseband input samples through the upconverter.

    Example:
    --------
    >>> upconv = Upconverter(fc=10e3, fs=48e3, gain='auto')
    >>> output_samples = upconv.process(input_samples)
    """

    def __init__(self, fc: float, fs: float, gain: float | str = "auto"):
        """
        Initialize an upconverter with the given parameters.

        Parameters:
        -----------
        fc: float
            The carrier frequency of the upconverter.
        fs: float
            The sample rate of the upconverter.
        gain: float|string
            The gain of the upconverter; the input samples are multiplied by `gain`. when set to `'auto'`, the gain is automatically determined so that the maximum absolute value of output samples is 1.
        """
        # Initialize upconverter parameters
        self.fc = fc
        self.fs = fs
        if isinstance(gain, (float, int)) or gain == "auto":
            self.gain = gain
        else:
            raise ValueError("Invalid gain value. Must be 'auto' or a numeric type.")

        # phase is used to ensure that consecutive invocations of the upconverter maintain phase continuity
        self.phase = 0

    def __repr__(self) -> str:
        return f"Upconverter(fc={self.fc}, fs={self.fs}, gain={self.gain})"

    def process(self, input_samples: np.ndarray) -> np.ndarray:
        """
        Process the input samples through the upconverter.

        Parameters:
        -----------
        input_samples: np.ndarray
            The complex-valued baseband input samples to be upconverted.

        Returns:
        --------
        output_samples: np.ndarray
            The upconverted output samples.
        """
        import numpy as np

        # Determine gain if set to 'auto'
        if self.gain == "auto":
            max_val = np.max(np.abs(input_samples))
            gain = 1.0 / max_val if max_val != 0 else 1.0
        else:
            gain = self.gain

        # Generate the time vector
        t = np.arange(len(input_samples)) / self.fs

        # Perform upconversion
        output_samples = np.real(
            gain * input_samples * np.exp(1j * (2 * np.pi * self.fc * t + self.phase))
        )

        # Update phase for continuity
        self.phase += 2 * np.pi * self.fc * len(input_samples) / self.fs
        self.phase = np.mod(self.phase, 2 * np.pi)

        return output_samples  # type: ignore
