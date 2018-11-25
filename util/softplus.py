import math


def softplus(a):
    # SOFTPLUS calculates softplus as log(1+exp(a))
    ret = math.log1p(1 + math.exp(a))
    return ret