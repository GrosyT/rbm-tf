

# RBMEARLYSTOP applies early stopping if enabled
#   Internal function.
#   If earlystopping is enabled the function tests wether the
#   current performance is better than the best performance seen.
#   For classification RBM the validatio error is checked for non classification
#   RBM's the ratio of free energies is checked.


def rbmearlystopping(rbm, opts, earlystop, epoch):
    if rbm.early_stopping == 1 and epoch%opts.test_interval == 0:
        isbest = 0
        # for classification RBM's check if validation is better than current best
        if rbm.classRBM == 1 and earlystop['best_error'] > rbm.val_error[-1]:
            isbest = 1
            err = rbm.val_error[-1]

        # for generative RBM's check if the ratio is below 0
        elif rbm.classRBM == 0 and earlystop['best_error'] > rbm.energy_ratio[-1]:
            if rbm.energy_ratio[-1] > 0.99: # check for overfitting
                isbest = 1
                err = rbm.energy_ratio[-1]

        if isbest:
            earlystop['best_str'] = "***"
            earlystop['best_error'] = err
            earlystop['best_rbm'] = rbm
            earlystop['patience'] = rbm.patience
        else:
            earlystop['best_str'] = ""
            earlystop['patience'] = earlystop.patience - 1
    return earlystop