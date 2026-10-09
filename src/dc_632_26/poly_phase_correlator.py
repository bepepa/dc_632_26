import numpy as np

class PolyPhaseCorrelator:
    """Sequential Polyphase Correlator for signal synchronization.
    
    This class implements a sequential polyphase correlator for signal synchronization. It 
    processes the output of the (pulse) matched filter and computes the correlation with the known preamble sequence for each sampling phase. Sequential processing means that the correlator updates its output sample by sample, rather than computing the correlation over the entire signal at once. This enables real-time synchronization and permits transitioning to demodulation as soon as synchronization is achieved.

    Attributes:
    -----------
    preamble_symbols : np.ndarray
        The known preamble sequence used for correlation.
    num_phases : int
        The number of sampling phases for the polyphase correlator; this equals the oversampling factor of the received signal.
    n_phase : int
        The current sampling phase index.
    n_symbol : int
        The current symbol index.

    Methods:
    --------
    __init__(preamble_symbols, num_phases)
        Initialize the correlator with the given preamble and number of phases.
    reset()
        Reset the internal state of the correlator.
    update(matched_filter_output)
        Update the correlator with a new matched filter output sample and return the correlation and norm-squared.
    correlate()
        Compute the correlation for the current buffer state.
    sq_magnitude_mf_out()
        Compute the norm-squared of the matched filter outputs for the current phase.
    buffer(n_phase):
        Return the last `len(preamble_symbols)` samples for the specified phase, starting with the oldest sample.
    
    """
    def __init__(self, preamble_symbols, num_phases):
        self.preamble_symbols = preamble_symbols
        self.num_phases = num_phases

        ## internal state
        # ring buffer for each phase; stores the last `len(preamble_symbols)` samples
        self._buffer = np.zeros((num_phases, len(preamble_symbols)), dtype=complex)
        self.n_phase = -1
        self.n_symbol = -1

    def __repr__(self):
        return f"PolyPhaseCorrelator(preamble_symbols={self.preamble_symbols}, num_phases={self.num_phases})"

    def reset(self):
        """Reset the internal buffers of the correlator.
        
        Resets the internal state of the correlator, including the buffer, phase, and symbol counters. Invoke this function before starting a new synchronization process.
        """
        self._buffer.fill(0)
        self.n_phase = -1
        self.n_symbol = -1

    def update(self, matched_filter_output):
        """Update the inner state of the correlator with a new matched filter output sample.

        Parameters:
        -----------
        matched_filter_output : complex
            The new sample from the matched filter output.

        Returns:
        --------
        tuple
            A tuple containing the correlation value and the norm-squared of the matched filter outputs for the current phase.
        """
        # Advance the phase and symbol counters
        self.n_phase = (self.n_phase + 1) % self.num_phases
        if self.n_phase == 0:
            self.n_symbol = (self.n_symbol + 1)

        # ring buffer for each phase; new sample overwrites the oldest one in a circular manner
        self._buffer[self.n_phase, 
                     self.n_symbol % len(self.preamble_symbols)] = matched_filter_output
        
        # return the correlation resulting from new MF output
        return self.correlate(), self.sq_magnitude_mf_out()

    def update2(self, matched_filter_output):
        """Update the inner state of the correlator with a new matched filter output sample.

        Parameters:
        -----------
        matched_filter_output : complex
            The new sample from the matched filter output.

        Returns:
        --------
        tuple
            A tuple containing the correlation value and the norm-squared of the matched filter outputs for the current phase.
        """
        # Advance the phase and symbol counters
        self.n_phase = (self.n_phase + 1) % self.num_phases
        if self.n_phase == 0:
            self.n_symbol = (self.n_symbol + 1)

        # ring buffer for each phase; new sample overwrites the oldest one in a circular manner
        self._buffer[self.n_phase, 
                     self.n_symbol % len(self.preamble_symbols)] = matched_filter_output
        
        # return the correlation resulting from new MF output
        return self.normalize(), self.sq_magnitude_mf_out()

    def correlate(self):
        """Compute the correlation for the current buffer state.
        
        Returns:
        --------
        complex
            The correlation value after insertion of the newest matched filter output sample.
        """

        # the buffer contents is rotated; the position of the oldest sample in the buffer
        # is equal to (self.n_symbol + 1) % len(self.preamble_symbols). That position needs
        # to be circularly shifted to the front.
        n_oldest = (self.n_symbol + 1) % len(self.preamble_symbols)
        return np.sum(np.roll(self._buffer[self.n_phase, :], -n_oldest) * 
                      np.conj(self.preamble_symbols))

    def normalize(self):
        """Compute the correlation for the current buffer state.
        
        Returns:
        --------
        complex
            The correlation value after insertion of the newest matched filter output sample.
        """

        # the buffer contents is rotated; the position of the oldest sample in the buffer
        # is equal to (self.n_symbol + 1) % len(self.preamble_symbols). That position needs
        # to be circularly shifted to the front.
        n_oldest = (self.n_symbol + 1) % len(self.preamble_symbols)
        z = np.roll(self._buffer[self.n_phase, :], -n_oldest)
        num = np.abs(np.sum(z*np.conj(self.preamble_symbols)))**2
        norm = num / np.sum(np.abs(z)**2) / np.sum(np.abs(self.preamble_symbols)**2)
        return norm
        

    def sq_magnitude_mf_out(self):
        """compute the norm-squared of the matched filter outputs for the current phase.
        
        Returns:
        --------
        float
            The norm-squared of the matched filter outputs for the current phase.
        """
        return np.sum(np.abs(self._buffer[self.n_phase, :])**2)



    def buffer(self, n_phase:int):
        """Retrieve the contents of the buffer for the specified phase, with the oldest sample rotated to the front so that the buffer appears in chronological order.
        
        Parameters:
        -----------
        n_phase : int
            The phase index for which to retrieve the buffer contents.

        Returns:
        --------
        np.ndarray
            The buffer contents for the specified phase, with the oldest sample rotated to the front.
        """
        n_oldest = (self.n_symbol + 1) % len(self.preamble_symbols)

        return np.roll(self._buffer[self.n_phase, :], -n_oldest)


