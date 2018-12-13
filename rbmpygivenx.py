import numpy as np
import sys
import math
sys.path.insert(0, './util/')
from softplus import softplus

def rbmpygivenx(rbm, x, train_or_test):
    # RBMPYGIVENX calculates class probabilities [p(y|x)]
    # internal function
    #
    # Copyright (c) Sřren Sřnderby july 2014

    n_classes = rbm.d.shape[0]
    n_samples = x.shape[0]

    cwx = np.matmul(rbm.W, np.transpose(x))
    cwx = np.add(np.reshape(rbm.c, (50, 1)), cwx)



    # cwx = (np.matmul(rbm.W, np.transpose(x))) + rbm.c
    # cwx =

    # only apply dropout in training mode
    if train_or_test == 'train' and rbm.dropout_hidden > 0:
        cwx * rbm.hidden_mask

    # rbm.U = rbm.U[:,None,:]
    rbm.U2 = np.reshape(rbm.U,(50, 1, 12))  # o: rbm.U2 = np.concatenate((rbm.U[:, None, :], np.zeros((rbm.U.shape[0], 99, rbm.U.shape[1]))), axis=1)

    F = rbm.U2 + cwx[:, :, None]  # np.ndarray.transpose() or (rbm.U.transpose(0, 2, 1))

    rbm.zeros = np.zeros((n_samples, n_classes))
    class_log_prob = rbm.zeros  # -o: class_log_prob = rbm.zeros[n_samples,n_classes]
    for y in range(n_classes):
        softplus_F = softplus(F[:, :, y])
        class_log_prob[:, y] = np.sum(softplus_F, axis=0) + rbm.d[y]   # axis=1

        # o: class_log_prob[:, y] = sum(softplus(F[:, :, y])) + rbm.d[y]
        # o2: class_log_prob[:, y] = np.sum(softplus(F[:, :, y]), axis=0) + rbm.d[y]
    # normalize probabilities
    class_log_prob_amax = np.reshape((np.amax(class_log_prob, 1)), (100, 1))
    # :o class_log_prob_amax = np.reshape((np.amax(class_log_prob, 1)), (100, 1))
    class_prob = class_log_prob - class_log_prob_amax
    class_prob = np.exp(class_prob)
    #  o: class_prob = np.exp(class_log_prob - class_log_prob_amax)
    # class_log_prob - (np.amax(class_log_prob, 1))
    # o: class_prob = np.exp(np.subtract(class_log_prob, class_log_prob_amax))
    class_prob_sum = np.reshape(np.sum(class_prob, axis=1),(class_prob.shape[0],1))
    class_prob = np.divide(class_prob, class_prob_sum)

    return class_prob, F














