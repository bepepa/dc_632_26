
import numpy as np
from dc_632_26.estimator import Estimator
from dc_632_26.detector import ThresholdDetector
from dc_632_26.poly_phase_correlator import PolyPhaseCorrelator
from dc_632_26.demodulation import GenericSlicer


#Receiver class
# Enumerate state for the receiver finite state machine
SYNCH:int = 0
DEMOD:int = 1

class Receiver():
    def __init__(self, 
                 preamble_seq: np.ndarray,
                 fsT: int, 
                 slicer: GenericSlicer, # generic slicer configured with constellation
                 detection_threshold: float= 0.3,
                 detection_useNorm: bool=True,
                 debug: bool = True
                 ):
        # initial state of the receiver is synchronization
        self.state = SYNCH
        self.fsT=fsT
        self.slicer= slicer
        self.debug=debug
        self.detection_threshold=detection_threshold
        self.detection_useNorm=detection_useNorm
        self.preamble_seq = preamble_seq

        # use a dictionary to store and return the results of processing
        self.results = {}

        # initialize subsystems
        self.correlator = PolyPhaseCorrelator(self.preamble_seq, num_phases=fsT)
        #self.detector = ThresholdDetector(215,self.preamble_seq,False) # <-------  missing; initialization?
        num_symbols_preamble = len(self.preamble_seq)  
        self.detector = ThresholdDetector(
            num_phases=self.fsT, 
            num_symbols=num_symbols_preamble, # length of your preamble match
            threshold=self.detection_threshold, 
            useNorm=self.detection_useNorm,
            preamble_seq=preamble_seq
        )
        
        self.estimator = Estimator(self.preamble_seq,2) 


        # store configuration parameters
        self.preamble_seq = preamble_seq
        
        # parameters (for corrections) to be set by estimation
        self.sampling_offset = -1
        self.start_symbol = -1
        self.phasor = 1+0j
        self.frequency_offset_per_symbol = 0

    def process(self, mf_out: np.ndarray, n_syms: int):
        """process the MF output according to state of the receiver"""
    
        #Initialize variables and 2D arrays 
        num_phases = self.fsT 
        num_symbols = len(mf_out) // num_phases
        
        corrs = np.zeros((num_phases, num_symbols), dtype=complex)
        mags = np.zeros((num_phases, num_symbols), dtype=float)

        #Calculate the total number of individual matched filter samples to loop over
        total_samples=len(mf_out)
        self.state = SYNCH

        # sample-by-sample streaming processing loop 
        for sample_count in range(total_samples):
            # Pull the current matched filter sample out of the array
            sample = mf_out[sample_count]
            
            # Update the unmodifiable correlator class state
            if self.detector.useNorm is True:
                corr, mag_sq = self.correlator.update2(sample)
            else:
                corr, mag_sq = self.correlator.update(sample)
            
            # Derive current p and k grid , note this will step through corr by row and then cols
            k = sample_count // num_phases
            p = sample_count % num_phases
            
            # boundary check to prevent , may not need
            if k < num_symbols:
                corrs[p, k] = corr
                mags[p, k] = mag_sq
            
            # Check for detection threshold crossing and use the peak climbing
            detected, p_ind, k_ind, corr_val = self.detector.detect(corr, mag_sq, sample_count)
            
            if detected:
                # Store synchronization parameters into Receiver object state
                self.sampling_offset = p_ind #This is the sampling phase
                self.start_symbol = k_ind    #This is the symbol that indicates the end of the preamble buffer

                received_preamble_buffer = self.correlator.buffer(p_ind) # this uses the peak we found to retrieve buffer contents
        
                #Put results into dict form
                self.results['detected'] = True
                self.results['p_ind'] = p_ind
                self.results['k_ind'] = k_ind
                self.results['corr_val'] = complex(corr_val)
                self.results['received_preamble_buffer']=self.correlator.buffer(p_ind)

                if self.debug is True:
                    print("\n" + "=" * 60)
                    print("[DEBUG] Detection Results")
                    print("Internal process() results at end of SYNCH mode")
                    print(
                    f"Peak detected at grid position: "
                    f"[phase (row)={self.results['p_ind']}, symbol(col)={self.results['k_ind']}]"
                    )
                    print("=" * 60 + "\n")
        
                self.state=DEMOD
                assert self.state == DEMOD

                # Stop loop the sample count loop
                break
        
        
        #Estimator Stuff Starts here 

        self.sampling_offset = self.results['p_ind']
        self.start_symbol = self.results['k_ind'] 

        (self.frequency_offset_per_symbol,self.phasor, noise_variance) = self.estimator.estimate(self.results['received_preamble_buffer'])
        
        self.results['phasor'] = self.phasor
        self.results['frequency_offset_per_symbol'] = self.frequency_offset_per_symbol

        # transition to the next state (demodulation) after successful synchronization
        self.state = DEMOD

        assert self.state == DEMOD, f"Unexpected state after SYNCH: {self.state}"

        ## Corrections and demodulation
        # pull out the samples to be demodulated   
        start_idx = (self.start_symbol + 1) * self.fsT + self.sampling_offset
        demod_samples = mf_out[start_idx : start_idx + n_syms* self.fsT : self.fsT]
        if self.debug is True:
            print(f"[DEBUG] self.start_symbol,{self.start_symbol},self.fsT,{self.fsT},self.sampling_offset,{self.sampling_offset}")
            print(f"[DEBUG] start_idx,{start_idx},demod_samples len,{len(demod_samples)}")
            print(f"self.frequency_offset_per_symbol ,{self.frequency_offset_per_symbol}")       
            print(f"n_syms,{n_syms}")
            print(f"self.phasor",{self.phasor})

        # apply frequency and phase correction to the demodulated samples
      
        demod_samples_corrected = (demod_samples * 
                                              np.exp(-1j * (2*np.pi*self.frequency_offset_per_symbol * (len(self.preamble_seq)+np.arange(n_syms)))))
                
        # demod_samples_corrected = (demod_samples * 
        #                                 np.exp(-1j * (2*np.pi*.1/31) * np.arange(n_syms))) #test true df
        
        demod_samples_corrected = demod_samples_corrected * (1/self.phasor)
            ## Slicer
            # map the corrected demodulated samples to the nearest constellation points
            # assuming BPSK for simplicity; modify as needed for other modulation schemes
        
        rx_bits = self.slicer.slice_symbols(demod_samples_corrected)
        self.results['bits'] = rx_bits
        self.results['demod_samples_corrected']=demod_samples_corrected

        return self.results
        



    
    

