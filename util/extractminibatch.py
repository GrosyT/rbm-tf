def extractminibatch(kk, minibatch_num, batchsize, x, opts):
    #  EXTRACTMINIBATCH extract minibatch
    #   INPUTS
    #               kk    : random permuation of i.e   kk = randperm(n_samples);
    #     minibatchnum    : current minibatch
    #        batchsize    : minibatch size
    #                x    : data
    #             opts    ; opts struct
    batch_start = ((minibatch_num - 1) * batchsize + 1) - 1  # -o: batch_start = minibatch_num - 1 * batchsize + 1 problem with
    batch_end = minibatch_num * batchsize            # matlab index starting at 1 and python at 0
    n_samples = x.shape[0]
    if (batch_end + batchsize) <= n_samples:
        idx = kk[batch_start:batch_end]
        batch = x[idx, :]
    else:
        batch = x[kk[batch_start:], :]
    return batch
