import numpy as np

##RBMUPCLASSRBM calculate p(h=1|v) for for class rbm
# INPUTS
#   rbm        : A rbm struct
#   vis        : the activation of the visible layer
#   ey         : activation of the class labels
#   act_func   : the activation function, either @sigm or @sigmrnd
#
# OUTPUTS
#   act_hid    : the activation of the hidden layer
#
# see "A practical guide to training restricted Boltzmann machines" eqn 7
# act is the activation function. Currently either sigm or sigmrnd
#
#
# NOTATION
#    v  : all data given as      [n_samples   x #vis]
#   ey  : all data given as      [n_samples   x #n_classes]
#    W  : vis - hid weights      [ #hid       x #vis ]
#    U  : label - hid weights    [ #hid       x #n_classes ]
#    b  : bias of visible layer  [ #vis       x 1]
#    c  : bias of hidden layer   [ #hid       x 1]
#    d  : bias of label layer    [ #n_classes x 1]


def rbmupclassrbm(rbm, vis, ey, act_func):

    ey_rbm_U = np.matmul(ey, np.transpose(rbm.U))
    vis_rbm_W = np.matmul(vis, np.transpose(rbm.W))
    act_hid_inner_addition = np.add(np.transpose(rbm.c), vis_rbm_W)
    act_hid_outer_addition = np.add(act_hid_inner_addition, ey_rbm_U)
    # apply activation function
    act_hid = act_func(act_hid_outer_addition)
    return act_hid