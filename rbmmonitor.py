from rbmclassprobs import rbmclassprobs
from rbmfreeenergyratio import rbmfreeenergyratio
import sys
sys.path.insert(0, './util/')
from accuracy import accuracy

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
            train_err, train_om = rbm.err_func(train_probs, opts.y_train)
            # o: train_err, train_om = rbm.err_func(train_probs, opts.y_train)
            rbm.train_error.append(train_err)
            rbm.train_error_measures.append(train_om)

            if opts.x_val.any():
                val_probs = rbmclassprobs(rbm, opts.x_val, opts.batchsize)
                val_err, val_om = rbm.err_func(val_probs, opts.y_val)

                rbm.val_error.append(val_err)
                rbm.val_error_measures.append(val_om)
                val_err = str(rbm.val_error[-1])
            else:
                val_err = "NA"

            perf = "  | Tr: %5f - Val: %s" % (rbm.train_error[-1], val_err)

        # non class RBM calculate free energy ratio
        elif opts.x_val:
            x_s = x[x_samples, :]
            x_val_s = opts.x_val[val_samples, :]
            rbm_free_energy_ratio = rbmfreeenergyratio(rbm, x_s, x_val_s)
            rbm.energy_ratio.append()

    return perf, rbm
