import math
import numpy as np

def reference_log_stitching(t):
    t = np.asarray(t, dtype=float)
    if np.any(t < 1) or not np.isfinite(t).all():
        raise ValueError('positive finite audit times required')
    return math.log(math.pi ** 2 / 6) + 2 * np.log(np.maximum(1.0, np.log2(t)))
