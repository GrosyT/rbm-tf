import math
import numpy as np


def softplus(a):
    # SOFTPLUS calculates softplus as log(1+exp(a))
    ret = np.log(1 + np.exp(a))  # -o : ret = math.log1p(1 + math.exp(a))
    return ret
