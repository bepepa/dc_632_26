import numpy as np

class Estimator:
    """Parameter Estimation of the detected Preamble.

    This class implements methods to estimate the channel complex gain, 
    the frequency offset, and the noise variance of the detected preamble. 

    Attributes:
    -----------
    preamble : np.ndarray
        The known preamble symbols sequence used for correlation.
    div : int
        The number of segments the preamble is split into for frequency offset estimation.
    """
    
    def __init__(self, preamble, divisions):
        '''Initialize the estimator with the known preamble and the number of segments for frequency offset estimation '''
        self.preamble = preamble
        self.div = divisions

    def _innerProduct(self, rxd, pream):
        '''Compute the discrete inner product between a received segment and a known (preamble) segment'''
        return np.sum(rxd * np.conj(pream))

    def complexGainEstimator(self, Rn, sn):
        '''Estimate the complex channel gain (amplitude and phase) from a received segment and its corresponding known segment.'''
        numerator = self._innerProduct(Rn, sn)
        denominator = self._innerProduct(sn, sn)
        return numerator / denominator

    def frequencyOffset(self, Rn, sn):
            '''
            Returns frequency offset in normalized frequency: cycles/symbol
            Estimates the frequency offset by splitting the received and known sequences into `div` segments, 
            estimating one complex gain per segment, and fitting phase versus segment center using an average.
            '''

            symbols_per_segment = len(sn) // self.div
            freq_offset_estimates = []

            for segment_index in range(self.div - 1):
                start_index = segment_index * symbols_per_segment

                received_segment_a = Rn[start_index : start_index + symbols_per_segment]
                received_segment_b = Rn[start_index + symbols_per_segment : start_index + 2 * symbols_per_segment]

                known_segment_a = sn[start_index : start_index + symbols_per_segment]
                known_segment_b = sn[start_index + symbols_per_segment : start_index + 2 * symbols_per_segment]

                segment_gain_a = self.complexGainEstimator(received_segment_a, known_segment_a)
                segment_gain_b = self.complexGainEstimator(received_segment_b, known_segment_b)

                phase_change = np.angle(segment_gain_b * np.conj(segment_gain_a))
                delta_f = phase_change / (2 * np.pi * symbols_per_segment)

                freq_offset_estimates.append(delta_f)

            return np.mean(freq_offset_estimates)

    def noiseVariance(self, Rn, cpx_gain_est=None):
        '''
        Estimate the noise variance at the matched filter output using the residual error after
        frequency correction and complex gain estimation.
        '''
        sn = self.preamble

        if cpx_gain_est is None:
            cpx_gain_est = self.complexGainEstimator(Rn, sn)

        residual = Rn - cpx_gain_est * sn
        noise_var = np.mean(np.abs(residual) ** 2)

        return noise_var

    def estimate(self, Rn): # --> freq_offset, complex_amplitude

        freq_offset = self.frequencyOffset(Rn, sn=self.preamble)

        n = np.arange(len(Rn))
        Rn_corrected = Rn * np.exp(-1j*2*np.pi*freq_offset*n)

        complex_gain = self.complexGainEstimator(Rn_corrected, self.preamble)

        noise_variance = self.noiseVariance(Rn, cpx_gain_est=None)

        return freq_offset, complex_gain, noise_variance




        

    
