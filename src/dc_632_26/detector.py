import numpy as np

class ThresholdDetector:
    """_summary_

    Attributes
    ----------
    threshold : float
        Pre-determined detection threshold (calculated from ROC curve analysis)
    preamble_seq : np.ndarray
        The preamble sequence (complex symbols)
    useNorm : bool
        If true, use normalized inner product as the test statistic.
        Otherwise, use correlation.
    """
    def __init__(self, threshold: float, preamble_seq: np.ndarray, useNorm=False):
        self.threshold = threshold
        self.preamble_seq = preamble_seq
        self.useNorm = useNorm

        self.threshold_breached = False
        self.last_breached_val = 0.0

    def __repr__(self):
        return f"ThresholdDetector(threshold={self.threshold})"


    def detect(self, corr, mag_sq, n):
        """_summary_qq

        _extended_summary_

        Parameters
        ----------
        corr : complex np.ndarray
            The correlation between the received sequence and known preamble.
            2d: [n_phase, n_symbol]
        mag_sq : float
            The norm-squared of the received sequence
        n : _type_
            _description_

        Returns
        -------
        detected : bool
            True if the signal has been detected
        p_ind : int
            Phase index of the detection
        k_ind : int
            Symbol index of the detection
        corr_val : complex
            The correlation value at the detection. None if no detection.
        """

         # ========================================================
        # TODO: TEMPORARY DUMMY OVERRIDE FOR TESTING OTHER MODULES
        # ========================================================
        # detected = True
        # p_ind = 1
        # k_ind = 43
        # corr_val = 0.95 + 0*1j
        
        # print(f"DEBUG OVERRIDE: detected={detected}, p_ind={p_ind}, k_ind={k_ind}, corr_val={corr_val}")
        # return detected, p_ind, k_ind, corr_val
        # ========================================================

        if self.useNorm:
            test_statistic = np.abs(corr)**2 / mag_sq / np.linalg.norm(self.preamble_seq)**2
        else:
            test_statistic = np.abs(corr)**2

        print(f"test_statistic,{test_statistic}")

        detected=False
        if np.any(test_statistic > self.threshold):
            max_val = np.max(test_statistic)
            # Only consider it a detection if we are now on the falling edge
            if self.threshold_breached and (max_val < self.last_breached_val):
                detected = True
            else:
                detected = False
            self.threshold_breached = True
            self.last_breached_val = max_val
        else:
            self.threshold_breached = False
            detected=False

        if detected:
           m_inds = np.where(test_statistic > self.threshold)
           p_ind = np.argmax(test_statistic[m_inds[0], m_inds[1]])
           k_ind = m_inds[1][p_ind]
           corr_val = corr[p_ind, k_ind]
        else:
            p_ind = None
            k_ind = None
            corr_val = None

        print(f"detected,p_ind,k_ind,corr_val,{detected},{p_ind},{k_ind},{corr_val}")
        
        return detected, p_ind, k_ind, corr_val

