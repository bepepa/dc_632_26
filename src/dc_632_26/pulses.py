import numpy as np
import matplotlib.pyplot as plt

# Defining the rectangular pulse
def rectangular(num_samples):
    
    '''
    Parameters:
    ------------
    num_samples (int): number of samples
    
    Returns:
    ------------
    pulse (np.array): normalized array of samples according to the pulse shape
    '''
    
    # Creating the rectangular shape
    pulse = np.ones(num_samples)
    
    # Normalizing to unit energy
    return pulse / np.linalg.norm(pulse)

# Defining the triangular pulse
def triangular(num_samples):
    
    '''
    Parameters:
    ------------
    num_samples (int): number of samples
    
    Returns:
    ------------
    pulse (np.array): normalized array of samples according to the pulse shape
    '''
    
    # Time axis
    t = np.arange(num_samples) / num_samples
    
    # Creating the triangular shape
    pulse = np.piecewise(t,
                         [((0 <= t) & (t < 0.5)),
                          ((0.5 <= t) & (t < 1))],
                         [lambda t: t,
                          lambda t: 1 - t])
    
    # Normalizing to unit energy
    return pulse / np.linalg.norm(pulse)

# Defining the cosine squared pulse
def cosine_squared(num_samples):
    
    '''
    Parameters:
    ------------
    num_samples (int): number of samples
    
    Returns:
    ------------
    pulse (np.array): normalized array of samples according to the pulse shape
    '''
    
    # Time axis
    t = np.arange(num_samples) / num_samples
    
    # Creating the cosine squared shape
    pulse = 1 / 2 - 1 / 2 * np.cos(2 * np.pi * t)
    
    # Normalizing to unit energy
    return pulse / np.linalg.norm(pulse)

def half_sine(num_samples):
    time=np.linspace(0., 1, num_samples+1)
    time = time[0:-1]
    # Choose pulse type here, need to add other types esp square root raised 
    pulse = np.sqrt(2)*np.sin(np.pi*time); # sinum_samples):
    return pulse