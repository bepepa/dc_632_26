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
    #def __init__(self, threshold: float, preamble_seq: np.ndarray, useNorm=False):
    def __init__(self, num_phases: int, num_symbols: int, threshold: float, preamble_seq: np.ndarray, useNorm=False):
        #Receiver parameters
        self.num_phases = num_phases
        self.num_symbols = num_symbols
        # Input by user
        self.threshold = threshold
        self.preamble_seq = preamble_seq
        self.useNorm = useNorm

        #Tracking flags for handling the peak-climbing state machine
        self.threshold_breached = False  #Starts False because we haven't crossed the threshold yet
        self.last_breached_val = -1.0   # Set to a dummy low value so any real stat will beat it
        self.samples_since_breach = 0      # Counts up only after we cross the threshold

        #Initialize and will be used for saving the absolute best peak metrics we find
        self.best_p = None                 # Will store the phase index of the peak apex
        self.best_k = None                 # Will store the symbol index of the peak apex
        self.best_corr = None              # Will store the complex correlation value at the apex

    def __repr__(self):
        return f"ThresholdDetector(threshold={self.threshold})"


    def detect(self, corr, mag_sq, sample_count):
        """_summary

        Processes samples one by one. Tracks the rising edge. Looks for a falling edge to find the location 
        of the best peak.  If the signal is noisy or plateaus and it doesn't find a falling edge for a specified limit, it returns
        the location of the best peak so far. Uses either test statistic based on corr (self.usenorm=0) 
        or on normalized inner product (self.usenorm=1) 

        Parameters
        ----------
        corr : complex
            The scalar correlation value for the current matched filter sample,
            evaluated at the given sample_count.
            (Replaces the legacy 2D [n_phase, n_symbol] matrix approach 
            to support real-time streaming).)
        mag_sq : float
            The norm-squared of the received sequence
        sample_count : integer
            A counter to parse through the rows (sampling phases) and columns (symbols) of the full corr matrix
            Total number of samples processed so far.
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

        if self.useNorm and self.preamble_seq is not None:
            test_statistic = np.abs(corr)**2 / mag_sq / np.linalg.norm(self.preamble_seq)**2
        else:
            test_statistic = np.abs(corr)**2

        #Derive spatial grid coordinates solely from the sample count
        k = sample_count // self.num_phases  # Symbol index (column)
        p = sample_count % self.num_phases   # Phase index (row)

        #print(f"sample_count={sample_count} | test_statistic={test_statistic:.4f}")


        detected=False

        # Test statistic is above threshold
        if (test_statistic > self.threshold):
            if not self.threshold_breached:
                # First time crossing threshold. Initialize tracking variables
                self.threshold_breached = True
                self.last_breached_val = test_statistic
                self.best_p = p
                self.best_k = k
                self.best_corr = corr
                self.samples_since_breach = 0
            else:
                self.samples_since_breach += 1 # increment last since breach

                # Check if we found a strictly better/higher peak point
                if test_statistic > self.last_breached_val:
                    self.last_breached_val = test_statistic
                    self.best_p = p
                    self.best_k = k
                    self.best_corr = corr
                    self.samples_since_breach = 0  # Reset because peak moved forward

                # NOISE TIMEOUT: If it stays high for 2 full symbols past peak, force detection
                elif self.samples_since_breach >= (self.num_phases):
                    print("-> Noise safety timeout triggered. Locking in best peak found.")
                    detected = True
                    self.threshold_breached = False

                # CLEAN FALLING EDGE: A 10% drop might mean we passed it
                elif test_statistic < (self.last_breached_val * 0.9):
                    detected = True
                    self.threshold_breached = False

        #ELSE: 
        else:
            # If we were actively tracking a peak and suddenly drop off a steep cliff
            if self.threshold_breached:
                detected = True
            
            # Reset the gatekeeper flag because we are back in a quiet search mode
            self.threshold_breached = False

            
        if detected:
            p_ind=int(self.best_p)
            k_ind=int(self.best_k)
            corr_val=complex(self.best_corr)
            #print(f"Peak locked at sample {sample_count - self.samples_since_breach}")
        else:
            p_ind = None
            k_ind = None
            corr_val = None

        #print(f"detected,p_ind,k_ind,corr_val,{detected},{p_ind},{k_ind},{corr_val}")
        
        return detected, p_ind, k_ind, corr_val

