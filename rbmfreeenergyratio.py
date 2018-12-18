import numpy as np
import sys
sys.path.insert(0, './util/')
from softplus import softplus
# RBMFREEENERBYRATIO calculates the free energy ratio between x anx x_val
# see https://dl.dropboxusercontent.com/u/19557502/5_03_free_energy.pdf
#
# NOTATION
# data : all data given as [n_samples x #vis]
# v : all data given as [n_samples x #vis]
# ey : all data given as [n_samples x #n_classes]
# W : vis - hid weights [ #hid x #vis ]
# U : label - hid weights [ #hid x #n_classes ]
# b : bias of visible layer [ #vis x 1]
# c : bias of hidden layer [ #hid x 1]
# d : bias of label layer [ #n_classes x 1]
# Copyright Sřren Sřnderby july 2014


def freeenergy(rbm, x):
    # calculates free energy for all samples in x
    rbm_W_multi_xT = np.matmul(rbm.W, np.transpose(x))
    rbm_c_p_rbm_W_multi_xT = rbm.c + rbm_W_multi_xT
    wxc = softplus(rbm_c_p_rbm_W_multi_xT)
    F = - (np.matmul(np.transpose(rbm.b), np.transpose(x))) + np.sum(wxc, axis=0)
    return F


def rbmfreeenergyratio(rbm, x, x_val):
    assert x.shape[0] == x_val.shape[0]
    if rbm.classRBM == 1:
        raise ValueError("Not implemented for class RBM's")

    F_x = freeenergy(rbm, x)
    F_x_val = freeenergy(rbm, x_val)
    ratio = np.mean(F_x_val / F_x)
    return ratio

