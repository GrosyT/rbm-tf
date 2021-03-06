import numpy as np
import math
import scipy.io as sio
import sys
import statistics
sys.path.insert(0, './util/')
from extractminibatch import extractminibatch
from rbmsemisuplearn import rbmsemisuplearn
from rbmapplygrads import rbmapplygrads
from rbmmonitor import rbmmonitor
from rbmearlystopping import rbmearlystopping
from dbnsetup import create_func




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

    numbatches = n_samples / opts.batchsize
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
    chainsy = []
    best_str = ''

    if rbm.train_func == rbmsemisuplearn:
        semisup = 1
        l_semisup = 0
        n_samples_semisup = opts.x_semisup.shape[0]
        numbatches_semisup = int(n_samples_semisup / opts.batchsize)
        assert numbatches_semisup % 1 == 0, "Number of semisup batches is not integer"
    else:
        semisup = 0

    for epoch in range(opts.numepochs):
        kk = np.random.permutation(n_samples)
        if semisup:
            kk_semisup = np.random.permutation(n_samples_semisup)
        err = 0

        # in each epoch update rbm parameters
        rbm.curMomentum = rbm.momentum(epoch)  # rbm.momentum(epoch)
        rbm.curLR = rbm.learningrate(opts.t_learningrate[0], rbm.curMomentum)
        rbm.curCDn = rbm.cdn(epoch)

        for l in range(int(numbatches)):            # todo: check extractminibatch
            l_2 = l + 1
            v0 = extractminibatch(kk, l_2, opts.batchsize, x_train, opts)  # changed l cycle variable for correct index
            if rbm.classRBM == 1:
                ey = extractminibatch(kk, l_2, opts.batchsize, opts.y_train, opts)
            else:
                ey = []

            # iterate over semisup batches
            if semisup == 1:

                l_semisup = l_semisup + 1  # increment semisup batch
                if l_semisup > numbatches_semisup:
                    l_semisup = 1
                opts.x_semisup_batch = extractminibatch(kk_semisup, l_semisup, opts.batchsize,
                                                        opts.x_semisup, opts)
            if opts.traintype == 'PCD' and init_chains == 1:
                # init chains in first epoch if Persistent contrastive divergence

                # augment semisup PCD chains starting position
                if semisup:
                    # init semisup chains at mean training set values -o: not sure if that is correct?
                    # statistics.mean(opts.y_train)
                    meany = samplematrix(np.repeat(np.mean(opts.y_train, axis=0), opts.batchsize))
                    chains = np.concatenate((opts.x_semisup_batch,v0), axis=1)
                    chainsy = np.concatenate((meany,ey),axis=1)
                else:
                    chains = v0
                    chainsy = ey
                init_chains = 0

            if rbm.dropout_hidden > 0:
                rbm.hidden_mask = (rbm.rand(n_hidden.shape[0],opts.batchsize.shape[0])) > rbm.dropout_hidden

            # calculate rbm gradients
            grads, c_err, chains, chainsy = rbm.train_func(rbm, v0, ey, opts, chains, chainsy)

            err = err + c_err

            # update weights, LR,decay and momentum
            rbm = rbmapplygrads(rbm, grads, v0, ey, epoch)
        if len(rbm.error) == 0:
            rbm.error.append(err / numbatches)
        else:
            rbm.error.append(err / numbatches)
            #  O: rbm.error[-1] = err / numbatches

        # calc train\val performance.
        perf, rbm = rbmmonitor(rbm, x_train, opts, x_samples, val_samples, epoch)
        earlystop = rbmearlystopping(rbm, opts, earlystop, epoch)

        # stop training?
        if rbm.early_stopping and earlystop["patience"] < 0:
            print("No more Patience. Returning best RBM")
            earlystop["best_rbm"].val_error = rbm.val_error
            earlystop["best_rbm"].train_error = rbm.train_error
            earlystop["best_rbm"].error = rbm.error
            rbm = earlystop["best_rbm"]
            break

        # display output
        epochnr = " Epoch %s/%s." % (str(epoch + 1), str(opts.numepochs))
        avg_err = ' Avg. recon. err: ', str(err / numbatches)
        lr_mom = 'LR: ', str(rbm.curLR), '. Mom.: ', str(rbm.curMomentum)
        print(epochnr, avg_err, lr_mom, perf, earlystop["best_str"])

        if opts.outfile:
            if opts.early_stopping:
                best_rbm = earlystop["best_rbm"]
                sio.savemat('rbm', best_rbm)
                del best_rbm
            else:
                sio.savemat('rbm', rbm)
        if epoch == 15:
            debug_var = 1








    print("extractminibatch call check with shape: ", v0.shape)
    print("rbm.W - rbmtrain.py - súlyok az első rbm-ben: ", rbm.W.shape)
    # n_hidden, n_visible = rbmlist.W.shape

    # print("rbmlist at rbmtrain: ", rbmlist[0].classRBM)
    # print("rbmlist at rbmtrain: ", rbmlist)
    # print("rbmlist[u]classrbm at rbmtrain: ", rbmlist[0].classRBM)
    # rbmlist[0].cdn = 0

    return rbm
    # pass
