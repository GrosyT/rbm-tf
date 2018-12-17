import numpy as np
# RBMAPPLYGRADS applies momentum and learningrate and updates rbm weights
# Internal function used by rbmtrain
#
#   INPUT
#       rbm        : rbm struct
#       opts       : opts struct
#       grads.dw   : w weights chainge normalized by minibatch size
#       grads.db   : bias of visible layer weight change norm by minibatch size
#       grads.dc   : bias of hidden layer weight change norm by minibatch size
#       grads.du   : class label layer weight change norm by minibatch size
#       grads.dd   : class label hidden bias weight change norm by minibatch size
#       x          : current minibatch
#       ey         : if classRBM one hot encoded class labels otherwise empty
#       epoch      : current epoch number
#
#   OUTPUT
#       rbm     :  rbm struct with updated weights, LR and momentum

def rbmapplygrads(rbm, grads, x, ey, epoch):
    dw = grads['dw']
    db = grads['db']
    dc = grads['dc']
    dd = grads['dd']
    du = grads['du']

    # L2 regularization
    if rbm.L2 > 0:
        dw = dw - rbm.L2 * rbm.W
        if rbm.classRBM == 1:
            du = du - rbm.L2 * rbm.U

    # L1 regularization
    if rbm.L1 > 0:
        dw = dw - rbm.L2 * np.sign(rbm.W)
        if rbm.classRBM == 1:
            du = du - rbm.L2 * np.sign(rbm.U)

    if rbm.sparsity > 0:
        db = db - rbm.sparsity
        dc = dc - rbm.sparsity
        if rbm.classRBM == 1:
            dd = dd - rbm.sparsity

    # update weights and momentum of regular weights
    rbm.vW = rbm.curMomentum * rbm.vW + rbm.curLR * dw
    rbm.vc = rbm.curMomentum * rbm.vc + rbm.curLR * np.reshape(dc,rbm.vc.shape)
    if not(not db.any()):
        rbm.vb = rbm.curMomentum * rbm.vb + rbm.curLR * db

    rbm.W = rbm.W + rbm.vW
    rbm.b = rbm.b + rbm.vb
    rbm.c = rbm.c + rbm.vc

    # if classRBM update weights and momentum of U and d
    if rbm.classRBM == 1:
        rbm.vU = rbm.curMomentum * rbm.vU + rbm.curLR * du
        rbm.vd = rbm.curMomentum * rbm.vd + rbm.curLR * np.reshape(dd,rbm.vd.shape)
        rbm.U = rbm.U + rbm.vU
        rbm.d = rbm.d + rbm.vd

    # l2 norm constraint
    if rbm.L2 > 0:
        #rbm.W =
        pass # todo: implement l2 norm

    return rbm



