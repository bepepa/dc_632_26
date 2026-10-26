#!/usr/bin/env python3

# File: tracking.py
# Description: classes for tracking loops

"""Tracking loops module

Provides components and classes for implementing tracking loops, such as first-order and second-order loop filters, and integrators. The components are integrated to form complete tracking loop systems for phase and amplitude tracking.

Components:
-----------
    - First_Order_Filter: Implements a first-order loop filter with a single gain parameter.
    - Second_Order_Filter: Implements a second-order loop filter with linear and integrator gains.
    - Integrator: Implements a basic integrator for accumulating input samples over time.

Tracking Loops:
---------------
    The above building blocks are combined to provide tracking loops for phase and amplitude tracking of input signals.

    - PLL: Phase-Locked Loop for tracking the phase of input signals.
    - ALL: Amplitude-Locked Loop for tracking the amplitude of input signals.
"""

import numpy as np

#
# Components for tracking loops
#


class LoopFilter:
    """Base class for loop filters"""

    def __init__(self):
        pass

    def __call__(self, error: float) -> float:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def state(self) -> float:
        raise NotImplementedError("state not implemented")

    @state.setter
    def state(self, new_val: float):
        raise NotImplementedError("state setter not implemented")


class First_Order_Filter(LoopFilter):
    """Class representing a first-order loop filter

    Parameters
    ----------
    alpha : float
        loop gain of the first-order filter

    Methods:
    --------
    __call__(d_n: float) -> float
        Apply the first-order filter to the input sample.
    alpha (property)
        Get or set the loop gain of the filter.
    """

    def __init__(self, alpha: float):
        """Initialize the gain of the filter

        Input:
        alpha - loop gain
        """
        super().__init__()
        self._alpha = alpha

    def __call__(self, d_n: float) -> float:
        """invoke the filter

        Input:
        ------
        d_n: float
          single sample, representing error

        Returns:
        -------
        float
            filtered sample; type is the same as the input

        """
        return self._alpha * d_n

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, new_val: float):
        if new_val < 2 and new_val > 0:
            self._alpha = new_val
        else:
            raise ValueError("alpha = {:} makes PLL unstable".format(new_val))


class Second_Order_Filter(LoopFilter):
    """Class representing a second-order loop filter

    Parameters
    ----------
    alpha1 : float
        linear gain of the filter
    alpha2 : float
        integrator gain of the filter
    state : float, optional
        value held by integrator (default: 0)

    Methods
    -------
    __call__(d_n: float) -> float
        Apply the second-order filter to the input sample.
    alpha1 (property)
        Get or set the linear gain of the filter.
    alpha2 (property)
        Get or set the integrator gain of the filter.
    state (property)
        Get or set the state of the integrator; used to initialize frequency offset
    damping_factor (property)
        Compute the damping factor of the filter.
    """

    def __init__(self, alpha1: float, alpha2: float, state: float = 0):
        """Initialize the gains and the state of the filter

        Input:
        alpha1 - linear gain
        alpha2 - integrator gain
        state - value held by integrator (optional, default: 0)
        """
        super().__init__()
        self._alpha1: float = alpha1
        self._alpha2: float = alpha2
        self._state: float = state

    def __call__(self, d_n: float) -> float:
        """invoke the filter

        Input:
        ------
        d_n: float
            single sample, representing error

        Returns:
        -------
        float
            filtered sample; type is the same as the input
        """
        out = self._alpha1 * d_n + self._state
        self._state += self._alpha2 * d_n  # update state

        return out

    # below, we check that 0 <= alpha2 < alpha1 < 1
    @property
    def alpha1(self):
        return self._alpha1

    @alpha1.setter
    def alpha1(self, new_val: float):
        if new_val < 1 and new_val > 0:
            self._alpha1 = new_val
        else:
            raise ValueError("alpha1 = {:} can make PLL unstable".format(new_val))

    @property
    def alpha2(self):
        return self._alpha2

    @alpha2.setter
    def alpha2(self, new_val: float):
        if new_val < self._alpha1 and new_val > 0:
            self._alpha2 = new_val
        else:
            raise ValueError("alpha2 = {:} can make PLL unstable".format(new_val))

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_val: float):
        self._state = new_val

    @property
    def damping_factor(self) -> float:
        return self.alpha1 / 2 / np.sqrt(self.alpha2)

    @damping_factor.setter
    def damping_factor(self, new_val: float):
        if new_val <= 0:
            raise ValueError("damping_factor must be positive")

        self._alpha2 = self.alpha1**2 / (4 * new_val)


class Integrator:
    """Class representing an integrator

    Parameters
    ----------
    state : float, optional
        value held by the integrator (default: 0)

    Methods:
    -------
    __call__(x_n: float) -> float
        Apply the integrator to the input sample.
    state (property)
        Get or set the state of the integrator; used to initialize initial phase or amplitude offset.
    """

    def __init__(self, state: float = 0):
        self.state = state

    def __call__(self, x_n: float) -> float:
        "Compute the output from the integrator"
        out = self.state
        self.state += x_n

        return out

    @property
    def state(self) -> float:
        return self._state

    @state.setter
    def state(self, new_val: float):
        self._state = new_val


#
# Tracking loops
#
class TrackingLoop:
    """Base class for tracking loops (e.g., PLL, FLL)

    Methods
    -------
    correct(Z_n: complex) -> complex
        Apply the tracking loop correction to the input sample.
    update(Z_corrected: complex, Z_reference: complex)
        Update the tracking loop with the corrected and reference samples.
    """

    def correct(self, Z_n: complex) -> complex:
        raise NotImplementedError

    def update(self, Z_corrected: complex, Z_reference: complex):
        raise NotImplementedError


class PLL(TrackingLoop):
    """Class representing a phase-locked loop (PLL)

    Parameters
    ----------
    filter : LoopFilter, optional
        Loop filter used in the PLL (default: SecondOrderLoopFilter)

    Methods
    -------
    correct(Z_n: complex) -> complex
        Apply the tracking loop correction to the input sample.
    update(Z_corrected: complex, Z_reference: complex)
        Update the tracking loop with the corrected and reference samples.
    """

    def __init__(
        self, filter: LoopFilter = Second_Order_Filter(alpha1=0.1, alpha2=0.0025)
    ):
        self.filter = filter
        self.integrator = Integrator()

    def correct(self, Z_n: complex) -> complex:
        """Correct the phase of the complex input sample using the PLL

        Input
        -----
        Z_n : complex
            Complex input sample to be corrected; from matched filter output

        Returns
        -------
        complex
            Phase-corrected complex output sample; suitable as input to the slicer
        """
        # the phase correction comes from the integrator state
        return Z_n * np.exp(-1j * self.integrator.state)

    def update(self, Z_corrected: complex, Z_reference: complex):
        """Update the PLL with the given phase error

        Input
        -----
        Z_corrected : complex
            Phase-corrected complex input sample; output of the PLL correction stage
        Z_reference : complex
            Reference complex input sample; either known a priori or derived from the transmitted signal

        """
        # compute phase error between corrected and reference signals
        phase_error = np.angle(Z_corrected * np.conj(Z_reference))

        # pass through the loop filter
        correction = self.filter(phase_error)

        # and then to integrator
        self.integrator(correction)

    def set_phase(self, phase: float):
        """Set the phase of the PLL integrator directly

        Input
        -----
        phase : float
            Phase value to set the integrator to
        """
        self.integrator.state = phase

    def set_frequency(self, frequency: float):
        """Set the (normalized) frequency of the PLL integrator directly

        Input
        -----
        frequency : float
            Frequency value, normalized with respect to the symbol rate
        """
        try:
            self.filter.state = 2 * np.pi * frequency
        except NotImplementedError:
            print("Warning: Loop filter does not support setting frequency")


class ALL(TrackingLoop):
    """Class representing an amplitude-locked loop (ALL)

    Parameters
    ----------
    filter : LoopFilter, optional
        Loop filter used in the ALL (default: First_Order_Filter)

    Methods
    -------
    correct(Z_n: complex) -> complex
        Apply the tracking loop correction to the input sample.
    update(Z_corrected: complex, Z_reference: complex)
        Update the tracking loop with the corrected and reference samples.

    """

    def __init__(self, filter: LoopFilter = First_Order_Filter(alpha=0.1)):
        self.filter = filter
        self.integrator = Integrator()

    def correct(self, Z_n: complex) -> complex:
        """Correct the amplitude of the complex input sample using the ALL

        Input
        -----
        Z_n : complex
            Complex input sample to be corrected; from matched filter output

        Returns
        -------
        complex
            Amplitude-corrected complex output sample; suitable as input to the slicer
        """
        # the gain (~1/|Z_n|) correction comes from the integrator state
        return Z_n * self.integrator.state

    def update(self, Z_corrected: complex, Z_reference: complex):
        """Update the ALL with the given amplitude error

        Input
        -----
        Z_corrected : complex
            Amplitude-corrected complex input sample; output of the ALL correction stage
        Z_reference : complex
            Reference complex input sample; either known a priori or derived from the transmitted signal

        """
        # compute amplitude error between corrected and reference signals
        amplitude_error = 1 - np.abs(Z_corrected / Z_reference)

        # pass through the loop filter
        correction = self.filter(amplitude_error)

        # and then to integrator
        self.integrator(correction)

    def set_gain(self, gain: float):
        """Set the ALL integrator to achieve the desired gain

        Input
        -----
        gain : float
            Desired gain
        """
        self.integrator.state = 1 / gain
