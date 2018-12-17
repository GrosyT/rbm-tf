from rbmclassprobs import rbmclassprobs

# RBMMONITOR monitor rbm performance
# Internal function used by RBMTRAIN
# If the RBM is a classRBM the function augments rbm.train_perf and
# rbm.val_perf with the newest accuracies.
# If generativeRBM calcualte the free energy ratio and update rbm.energy_ratio
#
# val_samples is a vector of the same length as the number of  validation
# samples. Used when free energies are compared.
#
# INPUTS:
#         rbm : a rbm struct
#           x : the initial state of the hidden units
#        opts : opts struct
# val_samples : sample numbers in validatio set to be used for calculation
#               of free energies.
#     epoch   : current epoch number


def rbmmonitor(rbm, x, opts, x_samples, val_samples, epoch):

    perf = '.'
    if epoch%opts.test_interval == 0:
        if opts.classRBM:
            train_probs = rbmclassprobs(rbm, x, opts.batchsize)
