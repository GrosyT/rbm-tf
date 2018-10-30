import numpy as np


def rbmtrain(rbm, x_train, opts):

    # SETUP  and CHECKING
    # print("Type X_train: ",x_train.dtype)
    # print("Type X_train numpy: ", np.ndarray.dtype(x_train))
    if x_train.dtype != 'float32' and x_train.dtype != 'float64':
        raise ValueError("x must be float!")
    if 0 <= x_train.all() <= 1:
        print("x_train check okay")
    assert 0 <= x_train.all() <= 1, "X must be in range of [0,1]"  # 1 <= x_train.all() <= 2
    n_samples = x_train.shape[0]


    print("rbm.W - rbmtrain.py - súlyok az első rbm-ben: ", rbm.W.shape)
    # n_hidden, n_visible = rbmlist.W.shape

    # print("rbmlist at rbmtrain: ", rbmlist[0].classRBM)
    # print("rbmlist at rbmtrain: ", rbmlist)
    # print("rbmlist[u]classrbm at rbmtrain: ", rbmlist[0].classRBM)
    # rbmlist[0].cdn = 0

    return rbm
    # pass
