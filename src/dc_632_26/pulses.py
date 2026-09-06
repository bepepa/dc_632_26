import numpy as np
import matplotlib.pyplot as plt

# Defining the rectangular pulse
def rectangular(num_samples):
    
    return np.ones(num_samples) / num_samples

# Defining the triangular pulse
def triangular(num_samples):
    
    t = np.arange(num_samples) / num_samples
    
    pulse = np.sqrt(12) * np.piecewise(t,
                                   [((0 <= t) & (t < 0.5)),
                                    ((0.5 <= t) & (t < 1))],
                                   [lambda t: t,
                                    lambda t: 1 - t])
    
    return pulse

print(np.mean(triangular(24) ** 2))
print(np.mean(triangular(23) ** 2))
print(np.mean(triangular(25) ** 2))
print(np.mean(triangular(100) ** 2))
print(np.mean(triangular(3) ** 2))

# Defining the half sine pulse
def half_sine(num_samples):
    
    return

# Defining the consine squared pulse
def cosine_squared(num_samples):
    
    time_grid = np.arange(num_samples) / num_samples
    pulse = np.sqrt(8 / 3) * (1 / 2 - 1 / 2 * np.cos(2 * np.pi * time_grid))
    
    return pulse