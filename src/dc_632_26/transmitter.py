# this module contains a transmitter block for a digital communication system
"""Transmitter module for a digital communication system.

This module contains a transmitter block that performs modulation mapping, pulse shaping, and upconversion of digital signals.

It provides a class `Transmitter` that can be used to create a digital transmitter object with specified parameters.
"""

import numpy as np

from dc_632_26.pulses import Pulse
from dc_632_26.pulse_shaper import PulseShaper
from dc_632_26.constellations import Constellation
from dc_632_26.freq_shift import Upconverter
from dc_632_26.modulation import modulation


class Transmitter:
    """Class that represents a digital transmitter with modulation mapping, pulse shaping and upconversion.

    Parameters
    ----------
    fc : float
        Carrier frequency.
    fs : float
        Sampling frequency
    constellation : object
        Constellation object for modulation
    pulse_shape : object
        Pulse shaping object, configured with the desired pulse shape and the oversampling factor
    uc: Upconverter | None
        Upconverter object for frequency upconversion, or None if no upconversion is desired.

    Example:
    --------
    >>> from dc_632_26.constellations import standard_constellation
    >>> from dc_632_26.pulses import CosineSquaredPulse
    >>> from dc_632_26.pulse_shaper import PulseShaper
    >>> from dc_632_26.transmitter import Transmitter

    >>> fc = 10e3
    >>> fs = 44e3
    >>> fsT = 32

    >>> constellation = standard_constellation.QPSK
    >>> pulse = CosineSquaredPulse(fsT)
    >>> pulse_shaper = PulseShaper(pulse, fsT)
    >>> tx = Transmitter(fc, fs, constellation, pulse_shaper)
    """

    def __init__(
        self,
        fc: float,
        fs: float,
        constellation: Constellation,
        pulse_shaper: PulseShaper,
        uc: Upconverter | None = None,
    ):
        """Initialize the transmitter with carrier frequency, sampling frequency, constellation, pulse shaping function"""
        self.fc = fc
        self.fs = fs
        self.uc = uc
        self.constellation = constellation
        self.pulseShaper = pulse_shaper

        self.fsT = pulse_shaper.oversamp

    def __repr__(self) -> str:
        return f"Transmitter(fc={self.fc}, fs={self.fs}, constellation={self.constellation}, pulse_shaper={self.pulseShaper}, uc={self.uc})"

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Modulate the given bits into a passband signal using the transmitter's parameters.

        Parameters
        ----------
        bits : np.ndarray
            Array of input bits to be modulated.

        Returns
        -------
        np.ndarray
            Upconverted passband signal.
        """
        # map bits to symbols
        symbols = modulation(bits, self.constellation)

        # pulse shaping
        bb_signal = self.pulseShaper.generate_waveform(symbols)

        # upconvert
        if self.uc is not None:
            tx_signal = self.uc.process(bb_signal)
        else:
            tx_signal = bb_signal

        return tx_signal
