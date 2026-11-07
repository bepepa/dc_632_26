#!/usr/bin/env python3
# File: equalizer.py
# Equalizer functions for signals

import numpy as np


class LMS:
    """class that implements the LMS algorithm for a linear or DFE equalizer

    It has the following parameters:
    - Kf: length of feedforward filter
    - Kb: length of feedback filter, set to 0 for linear equalizer
    - mu: scaling factor for gradient descent (default: 0.5)

    The LMS class provides two methods:
    - update: computes updated weights based on the difference between the eualizer output and the symbol decision
    - equalize: computes the equalizer output
    both methods are invoked once per symbol period.

    The order of invocation must be: LMS.equalize -> decision -> LMS.update

    The class maintains a vector that stores Kb prior decisions. Theis state
    vector is updated in the `update` method. Therefore, the update method MUST be called for every symbol
    even if the equalizer weights should not be updated (set `update_weights=False` when calling `update` to
    prevent that weights are recomputed).
    """

    def __init__(self, Kf: int, Kb: int, mu: float = 0.5) -> None:
        self.Kf = Kf  # length of FF filter
        self.Kb = Kb  # <-- length of FB filter
        self.mu = mu  # forgetting factor

        self.v = np.zeros(Kf + Kb, dtype=complex)
        self.alpha = 0.1  # small positive constant

        self.sr_sym = np.zeros(Kb, dtype=complex)  # <-- store recent decisions

    def update(
        self,
        error: complex,
        sym: complex,
        vec_r: np.ndarray,
        update_weights: bool = True,
    ) -> None:
        """LMS update

        Inputs:
        - error: difference between training/decision and equalizer output
        - sym: new training symbol or decision
        - vec_r: vector of received samples
        - update_weights: flag indicating if equalizer weights are recomputed (default: True)

        Returns:
        nothing

        Updates:
        - the vectors containing recent training/decision symbols as well as recent received samples are updated
        - if `update_weights is True` then the weight vector is recomputed
        """
        if update_weights:
            # error
            e = error  # sym - self.equalize(vec_r)

            # update weight vector
            scale = self.mu / (self.alpha + vec_r.conj() @ vec_r)
            # scale = self.mu
            self.v += np.concatenate((vec_r, self.sr_sym)) * np.conj(e) * scale

        if self.Kb > 0:
            # update state vector for a DFE
            self.sr_sym[:] = np.concatenate(
                (self.sr_sym[1:], [sym])
            )  # <-- store new symbol

    def equalize(self, vec_r: np.ndarray):
        """compute equalizer output"""

        tmp = np.concatenate((vec_r, self.sr_sym))  # <--- work BOTH filters
        return (self.v.conj() @ tmp).squeeze()


class RLS:
    """class for carrying out the RLS algorithm

    This class implements the RLS algorithm for a linear or DFE equalizer.

    It has the following parameters:
    - Kf: length of feedforward filter
    - Kb: length of feedback filter, set to 0 for linear equalizer
    - fsT: oversampling factor for FF filter
    - lam: forgetting factor (0 < lambda <= 1)

    The RLS class provides two methods:
    - update: computes updated weights based on the difference between the equalizer output and the symbol decision
    - equalize: computes the equalizer output
    both methods are invoked once per symbol period.

    The order of invocation must be: RLS.equalize() -> decision() -> RLS.update()

    The class maintains a vector that stores Kb prior decisions. This state
    vector is updated in the `update` method. Therefore, the update method MUST be called for every symbol
    even if the equalizer weights should not be updated (set `update_weights=False` when calling `update` to
    prevent that weights are recomputed).
    """

    def __init__(self, Kf: int, Kb: int, fsT: int, lam: float = 1.0) -> None:
        self.Kf = Kf  # length of FF filter
        self.Kb = Kb  # length of FB filter
        self.M = fsT  # oversampling factor for FF filter
        self.lam = lam  # forgetting factor

        self.R_inv = 1 * np.eye(
            Kf + Kb, dtype=complex
        )  # inverse of auto-correlation matrix
        self.v = np.zeros(Kf + Kb, dtype=complex)  # weight vector

        ## shift register holds both recent samples and recent training symbols (or decsions)
        # the received samples are stored in the first Kf positions, the symbols are stored in the
        # final Kb positions
        self.sr_sym = np.zeros(Kb, dtype=complex)

    def update(
        self,
        error: complex,
        sym: complex,
        vec_rr: np.ndarray,
        update_weights: bool = True,
    ) -> None:
        """RLS recursion

        Inputs:
        - error - difference between training/decision and equalizer output
        - sym - the most recent training symbol or decision
        - vec_rr - vector of received samples
        - update_weights: set to False to prevent updating of equalizer weights (default: True)

        Returns:
        nothing

        Updates:
        - if `update_weights is True` all state variables are updated with the RLS algorithm
        - state vector holding prior training/decision symbols is always updated

        """
        if update_weights:
            # error
            e = error  # sym - self.equalize(vec_rr)

            # compute R_inv * vec_r
            tmp = np.concatenate((vec_rr, self.sr_sym))
            vec_g = self.R_inv @ np.expand_dims(tmp, 1)

            # update R_inv
            denom = self.lam + tmp.conj() @ vec_g
            self.R_inv = 1 / self.lam * (self.R_inv - vec_g @ vec_g.conj().T / denom)

            # update weight vector
            self.v += np.squeeze(vec_g) * np.conj(e) / denom

        # update the shift register that holds decisions
        if self.Kb > 0:
            self.sr_sym[:] = np.concatenate((self.sr_sym[1:], [sym]))

    def equalize(self, vec_rr):
        """compute equalizer output

        Inputs:
        vec_rr - vector of samples

        Returns:
        - equalizer output

        Updates:
        the state of the equalizer is unchanged
        """
        tmp = np.concatenate((vec_rr, self.sr_sym))

        return (self.v.conj() @ tmp).squeeze()
