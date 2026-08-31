import numpy as np

def cosine_squared(num_samples):
    
    time_grid = np.arange(num_samples) / num_samples
    pulse = 1 / 2 - 1 / 2 * np.cos(2 * np.pi * time_grid)
    
    return pulse