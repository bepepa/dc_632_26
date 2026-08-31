import numpy as np
import matplotlib.pyplot as plt

def rectangular(num_samples):
    
    return np.ones(num_samples)

def half_sine(num_samples):
    
    return

def cosine_squared(num_samples):
    
    time_grid = np.arange(num_samples) / num_samples
    pulse = np.sqrt(8 / 3) * (1 / 2 - 1 / 2 * np.cos(2 * np.pi * time_grid))
    
    return pulse