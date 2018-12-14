import numpy as np

def sigmrnd(P):
    ##SIGMRND returns neuron value as binary value with activation probability
    #     X = double(1./(1+exp(-P)))+1*randn(size(P));
    X = 1 / (1+np.exp(-P)) > np.random.rand(P.shape[0], P.shape[1])
    return X
