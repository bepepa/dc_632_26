# this module contains utility functions for a digital communication system
"""
Utility module for a digital communication system.

This module contains utility functions for our digital signals.

"""

import numpy as np

@staticmethod
def Qfunction(x):
    """
    Q function
    """
    from scipy.special import erfc
    return 0.5 * erfc(x / np.sqrt(2))