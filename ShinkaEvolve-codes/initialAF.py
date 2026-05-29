# EVOLVE-BLOCK-START
"""Coefficients construction"""

import numpy as np
import math

def construct_coefficients():
    """
    Construct the set of coefficients that occur at loop order L in the planar limit of form factor in the maximum super symmetric Yang Mills Theory.
    Returns:
        List of coefficients at loop order L
    """
    # Define the largest coefficient at L loop order
    L = 8
    Largest_coefficient = (4**L)/(2)*(math.factorial(2*(L-1)))/(math.factorial(L-1))
    Smallest_coefficient = -1*Largest_coefficient
    guess = np.arange(Smallest_coefficient,Largest_coefficient*2,Largest_coefficient)
    
    return 1, guess

# EVOLVE-BLOCK-END


