import sys
sys.path.insert(0, './util/')
from sigm import sigm
from sigmrnd import sigmrnd
from rbmdownx import rbmdownx

#RBMGENERATIVE calculate weight updates for generative RBM
# SEE sections contrastive divergence(CD) and persistent contrastive
# divergence (PCD), determined by opts.traintype
#
#   INPUTS:
#       rbm       : a rbm struct
#       v0        : the initial state of the hidden units
#       ey        : one hot encoded labels if classRBM otherwise empty
#       opts      : opts struct
#       chains    : current state of markov chains for visible units
#       chainsy   : current state of markov chains for label visible units
#       debug     : if it exists and is 1 save intermediate values to file
#                   currentworkdir/test_rbmgenerative.mat
#   OUTPUTS
#      A grads struct with the fields:
#       grads.dw   : w weights chainge normalized by minibatch size
#       grads.db   : bias of visible layer weight change norm by minibatch size
#       grads.dc   : bias of hidden layer weight change norm by minibatch size
#       grads.du   : class label layer weight change norm by minibatch size
#       grads.dd   : class label hidden bias weight change norm by minibatch size
#       curr_err   : current squared error normalized by minibatch size
#       chains     : updated value of chains for visible units
#       chainsy    : updated value of chains for label visible units.
#
#CONTRASTIVE DIVERGENCE (TYPE = 'CD')
# Normal contrastive divergence with k CD updates
#
# See
#   Hinton, G. (2002). Training Products of Experts by Minimizing Contrastive
#   Divergence. Neural Compu- tation, 14, 1771?1800.
#
#PERSISTENT CONTRASTIVE DIVERGENCE (TYPE = 'PCD')
#   The PCD approximation is obtained from the CD approximation by replacing the
#   sample v_k by a sample from a Gibbs chain that is independent from the
#   sample v_0 of of the training distribution. The algorithm corresponds to
#   standard CD learning without reinitializing the visible units of the Markov
#   chain with a training sample each time we want to draw a sample v_k
#   approximately from the RBM distribution. Instead one keeps ?persistent?
#   chains which are run for k Gibbs steps after each parameter update
#   (i.e., the initial state of the current Gibbs chain is equal to v_k from
#   the previous update step)
#
# see also
#     Tieleman, Tijmen.
#     "Training restricted Boltzmann machines using approximations to the
#     likelihood gradient."
#     Proceedings of the 25th international conference on Machine learning.
#     ACM, 2008.
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
# See also RBMDISCRMINATIVE RBMHYBRID RBMSEMISUPLEARN
#
# Copyright Sřren Sřnderby June 2014


def rbmgenerative(rbm, v0, ey, opts, chains, chainsy):

    train_type = opts.traintype # shadows built-in variable, matlab uses "type"

    # add dropout
    if rbm.dropout_hidden > 0:
        up = rbm.rbmup(rbm, vis, ey, act_func)*rbm.hidden_mask
    else:
        up = rbm.rbmup

    h0 = up(rbm, v0, ey, sigm)
    h0_rnd = (h0 > rbm.rand(h0.shape[0], h0.shape[1]))
    #h0_rnd.astype(float)

    # For contrastive divergence use the input vectors as starting point
    # for Persistent contrastive divergence we use the persistent chains as
    # starting point for the sampling
    if train_type == "CD":
        hid = h0_rnd
    elif train_type == "PCD":
        hid = up(rbm, chains, chainsy, sigmrnd)
    for n in range(rbm.curCDn - 1):
        visx = rbmdownx(rbm, hid, sigmrnd)
        visy = rbm.rbmdowny(rbm, hid)
        hid = up(rbm, visx, visy, sigmrnd)

    # in last up/down dont sample hidden because it introduces sampling noise








