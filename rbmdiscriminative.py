from rbmpygivenx import rbmpygivenx
import sys
import numpy as np
sys.path.insert(0, './util/')
from predict import predict
from samplematrix import samplematrix
from sigm import sigm
# RBMDISCRIMINATIVE calculate weight updates for discriminative RBM
#
#   INPUTS:
#       rbm       : a rbm struct
#       x        : the initial state of the hidden units
#       ey        : one hot encoded labels if classRBM otherwise empty
#       opts      : opts struct
#       chains    : not used, pass in anything
#       chainsy   : not used, pass in anything
#       debug     : if it exists and is 1 save intermediate values to file
#                   currentworkdir/test_rbmdiscriminative.mat
#
#   OUTPUTS
#      A grads struct with the fields:
#       grads.dw   : w weights chainge normalized by minibatch size
#       grads.db   : bias of visible layer weight change norm by minibatch size
#                    (db is zero for the discriminative RBM, returns [])
#       grads.dc   : bias of hidden layer weight change norm by minibatch size
#       grads.du   : class label layer weight change norm by minibatch size
#       grads.dd   : class label hidden bias weight change norm by minibatch size
#       curr_err   : not used, returns 0
#       chains     : not used, returns []
#       chainsy    : not used, returns []
#
#
#
# References
#    [1] H. Larochelle and Y. Bengio, ?Classification using discriminative
#        restricted Boltzmann machines,? ? 25th Int. Conf. Mach. ?, 2008.
#    [2] H. Larochelle and M. Mandel, ?Learning algorithms for the
#        classification restricted boltzmann machine,? J. Mach.  ?, 2012.
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
# See also RBMGENERATIVE RBMHYBRID RBMSEMISUPLEARN


def rbmdiscriminative(rbm, x, ey, opts, chains, chainsy, debug=None):
    n_classes = rbm.d.shape[0]

    p_y_given_x, F = rbmpygivenx(rbm, x, 'train')

    F_sigm = sigm(F)
    F_sigm_prob = np.zeros(F_sigm.shape)
    for c in range(n_classes):
        F_sigm_prob[:, :, c] = np.multiply(F_sigm[:, :, c], np.transpose(p_y_given_x[:, c]))
        # O: F_sigm_prob[:,:,c] = np.matmul(F_sigm[:,:,c], np.transpose(p_y_given_x[:,c]))

    # init grads
    dw = np.zeros(rbm.W.shape)   # :o dw = rbm.zeros((rbm.W.shape)
    du = np.zeros(rbm.U.shape)  # :o du = rbm.zeros(rbm.U.shape)
    dc = np.zeros(rbm.c.shape)  # :o dc = rbm.zeros(rbm.c.shape)

    class_labels = predict(ey)

    for c in range(n_classes):
        # dw grad

        # find idx for current class
        bin_idx = class_labels == c

        lin_idx = np.where(c == class_labels)

        a = np.matmul(F_sigm[:, lin_idx[0], c],x[lin_idx[0], :])  # # O: a = F_sigm[:, lin_idx, c]
        b = np.matmul(F_sigm_prob[:, :, c], x)
        dw = dw + a - b

        # du grad
        sum_F_sigm = np.sum(F_sigm[:,bin_idx,c], axis=1)
        sum_F_sigm_prob = np.sum(F_sigm_prob[:, :, c], axis=1)
        du[:, c] = sum_F_sigm - sum_F_sigm_prob

        # dc
        dc_diff = np.subtract(np.sum(F_sigm[:, bin_idx, c], axis=1), np.sum(F_sigm_prob[:, :, c], axis=1))
        dc = np.reshape(dc, (1, dc.shape[0])) + np.transpose(dc_diff)

        # dd grad
        dd = np.transpose(np.sum(ey - p_y_given_x, axis=0))

        # create grads struct and scale grad by minibatch size.
        grads = {
            "dw": dw / opts.batchsize,
            "db": np.zeros(rbm.b.shape),
            "dc": dc / opts.batchsize,
            "dd": dd / opts.batchsize,
            "du": du / opts.batchsize,
        }

        curr_err = 0
        chains = []
        chainsy = []

        return grads, curr_err, chains, chainsy

