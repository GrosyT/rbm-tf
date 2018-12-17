import numpy as np
##RBMDOWNYCLASSRBM calculates p(v_label = 1 | h) for label units in class RBM
#
# INPUTS
#   rbm           : A rbm struct
#   hid_act       : the activation of the hidden layer
#
# OUTPUTS
#   act_vis_y : The activation of the class label visible units
#
# see "A practical guide to training restricted Boltzmann machines" eqn 8
# act is the activation function. currently either sigm or sigmrnd
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
# See also RBMDOWNX RBMUP SAMPLEMATRIX


def rbmdownyclassrbm(rbm, hid_act):
    hid_act_rbm_U = np.matmul(hid_act, rbm.U)
    rbm_d_add_hid_act_rbm_U = np.add(np.transpose(rbm.d), hid_act_rbm_U)
    exp_rbm_d_add_hid_act_rbm_U = np.exp(rbm_d_add_hid_act_rbm_U)
    sum_act_vis_y = np.sum(exp_rbm_d_add_hid_act_rbm_U, axis=1)
    sum_act_vis_y = np.reshape(sum_act_vis_y, (100, 1))
    act_vis_y = exp_rbm_d_add_hid_act_rbm_U / sum_act_vis_y
    return act_vis_y
