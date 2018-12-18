import sys
sys.path.insert(0, './util/')
from chunkify import chunkify
from inspect import signature
from rbmpygivenx import rbmpygivenx
import numpy as np


# RBMCLASSPROBS calculate class probabilities for a classification RBM
#
#  INPUTS
#   rbm       : A rbm struct
#          x  : matrix of samples  (n_samlples-by-n_features)
#   batchsize : optionally takes a minibatch size in which case the result
#               is calculated in minibatches to save memory
#
#  OUTPUT
#   class_prob_res : class probabilites for each class (n_samples-by-n_classes)
#
#  NOTES
#   see equation 2 of the paper:
#   Learning algorithms for the classification restricted boltzmann machine
#
#
# NOTATION
# data  : all data given as      [n_samples   x #vis]
#    v  : all data given as      [n_samples   x #vis]
#   ey  : all data given as      [n_samples   x #n_classes]
#    W  : vis - hid weights      [ #hid       x #vis ]
#    U  : label - hid weights    [ #hid       x #n_classes ]
#    b  : bias of visible layer  [ #vis       x 1]
#    c  : bias of hidden layer   [ #hid       x 1]
#    d  : bias of label layer    [ #n_classes x 1]
#
# See also DBNCLASSPROBS
#
# Copyright(c) Sřren Sřnderby july 2014


def rbmclassprobs(rbm, x, batchsize):
    n_visible = rbm.W.shape[1]
    if not rbm.classRBM:
        raise ValueError("Class probabilities can only be calc. for classification RBM´s")
    if x.shape[1] != n_visible:
        raise ValueError("x has wrong dimensions")

    n_samples = x.shape[0]

    # check if result should be calculated in batches
    sig = signature(rbmclassprobs)
    if len(sig.parameters) == 3:
        numbatches = n_samples / batchsize
        assert numbatches%1 == 0, "numbatches not integer"
        #
        chunks = chunkify(batchsize, x)
    else:
        chunks = chunkify(n_samples, x)

    class_prob_res = []
    class_prob_res = np.empty((0, rbm.U.shape[1])) #np.array(class_prob_res)
    # :O class_prob_res = np.empty((batchsize,rbm.U.shape[1]))
    for i in range(len(chunks)):
        minibatch = x[chunks[i]['start']:chunks[i]['end'],:]
        class_prob, _ = rbmpygivenx(rbm, minibatch, 'test')
        class_prob_res = np.append(class_prob_res, class_prob, axis=0)
        # class_prob_res = np.concatenate((class_prob_res, class_prob), axis=0)

    return class_prob_res
