"""Fast and/or safe math routines.

Converted from Joel Veness's CTS implementation."""

import numpy as np
import math

def vector_log_add(log_x, log_y):
    """Given log x and log y, returns log(x + y)."""
    # Swap variables so log_y is larger.
    indices = np.nonzero(log_x > log_y)[0]
    log_x[indices], log_y[indices] = log_y[indices], log_x[indices]

    # Use the log(1 + e^p) trick to compute this efficiently
    # If the difference is large enough, this is effectively log y.
    delta = log_y - log_x
    indices1 = np.nonzero(delta<=50)[0]
    indices2 = np.nonzero(delta>50)[0]
    ret = np.zeros(delta.shape) 
    ret[indices1] = np.log1p(np.exp(delta)) + log_x
    ret[indices2] = log_y
    return ret

def log_add(log_x, log_y):
    """Given log x and log y, returns log(x + y)."""
    # Swap variables so log_y is larger.
    if log_x > log_y:
        log_x, log_y = log_y, log_x

    # Use the log(1 + e^p) trick to compute this efficiently
    # If the difference is large enough, this is effectively log y.
    delta = log_y - log_x
    return math.log1p(math.exp(delta)) + log_x if delta <= 50.0 else log_y
