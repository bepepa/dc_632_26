import numpy as np
import matplotlib.pyplot as plt

def rectangular(num_samples):
    
    return np.ones(num_samples)



def cosine_squared(num_samples):
    
    time_grid = np.arange(num_samples) / num_samples
    pulse = np.sqrt(8 / 3) * (1 / 2 - 1 / 2 * np.cos(2 * np.pi * time_grid))
    
    return pulse

def half_sine(num_samples):
    time=np.linspace(0., 1, num_samples+1)
    time = time[0:-1]
    # Choose pulse type here, need to add other types esp square root raised 
    pulse = np.sqrt(2)*np.sin(np.pi*time); # sinum_samples):
    return pulse