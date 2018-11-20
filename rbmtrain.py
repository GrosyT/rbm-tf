import numpy as np
import math


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
    n_hidden, n_visible = rbm.W.shape

    num_batches = n_samples / opts.batchsize
    assert n_samples % opts.batchsize == 0, "Number of batches is not integer"  # 1 <= x_train.all() <= 2

    # use validation set or not in calculation of free energy
    if opts.x_val == []:
        val_samples = []
        x_samples = []
        # print("if not validation debug false branch: ")
    else:
        n_val_samples = opts.x_val.shape[0]
        samples = np.random.permutation(x_train.shape[0])
        # if size of val set is larger than train set use trainset size otherwise
        # use size of validation set
        size_val_sample = n_val_samples if n_samples>=n_val_samples else n_samples
        x_samples = samples[0:size_val_sample]
        val_samples = np.arange(size_val_sample)

        # print("if not validation debug true branch: ")

    earlystop = dict(best_error=math.inf, patience=rbm.patience, best_str='')

    # RUN epochs
    init_chains = 1
    chains = []
    chainssy = []
    best_str = ''

    if rbm.train_func == 'rbmsemisuplearn':
        semisup = 1
        l_semisup = 0
        n_samples_semisup = opts.x_semisup.shape[0]
        numbatches_semisup = n_samples_semisup / opts.batchsize
        assert numbatches_semisup % opts.batchsize == 0, "Number of semisup batches is not integer"
    else:
        semisup = 0

    for epoch in range(opts.numepochs):
        kk = np.random.permutation(n_samples)
        if semisup:
            kk_semisup = np.random.permutation(n_samples_semisup)
        err = 0

        # in each epoch update rbm parameters
        rbm.curMomentum = rbm.momentum(opts.t_momentum[0])  # rbm.momentum(epoch)
        rbm.curLR = rbm.learningrate(opts.t_learningrate[0], rbm.curMomentum)
        rbm.curCDn = rbm.cdn(epoch)





    print("rbm.W - rbmtrain.py - súlyok az első rbm-ben: ", rbm.W.shape)
    # n_hidden, n_visible = rbmlist.W.shape

    # print("rbmlist at rbmtrain: ", rbmlist[0].classRBM)
    # print("rbmlist at rbmtrain: ", rbmlist)
    # print("rbmlist[u]classrbm at rbmtrain: ", rbmlist[0].classRBM)
    # rbmlist[0].cdn = 0

    return rbm
    # pass
