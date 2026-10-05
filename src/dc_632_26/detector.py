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
    def __init__(self, threshold: float, preamble_seq: np.ndarray, useNorm=True):
        self.threshold = threshold
        self.preamble_seq = preamble_seq
        self.useNorm = useNorm
        self.last_test_val = None

    def __repr__(self):
        return f"ThresholdDetector(threshold={self.threshold})"


    def detect(self, corr, mag_sq, n):
        """_summary_

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
        n_phase : int
            The detected sampling phase. None if no detection.
        corr_val : complex
            The correlation value at the detected sampling phase. None if no detection.
        """
        if self.useNorm:
            test_statistic = np.abs(corr)**2 / mag_sq / np.linalg.norm(self.preamble_seq)**2
        else:
            test_statistic = np.abs(corr)**2
        detected = np.any(test_statistic > self.threshold)
        if detected:
            m_inds = np.where(np.abs(test_statistic) > self.threshold)
            p_ind = np.argmax(np.abs(test_statistic[m_inds[0], m_inds[1]]))
            k_ind = m_inds[1][p_ind]
        else:
            p_ind = 0
            k_ind = 0
            n_phase = np.nan
            corr_val = np.nan

        self.last_test_val = test_statistic
        return detected, k_ind, corr[p_ind, k_ind]

