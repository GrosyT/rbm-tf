import numpy as np


def sigm(P):
    # SIGM sigmoid function
    X = 1 / (1+np.exp(-P))
    return X

#
# sigmtest = np.random.rand(5,5)
# print("SIGM test: \n", sigm(sigmtest))