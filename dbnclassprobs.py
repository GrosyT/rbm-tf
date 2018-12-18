from rbmclassprobs import rbmclassprobs
import sys
sys.path.insert(0, './util/')
from sigm import sigm

# function  class_probs = dbnclassprobs( dbn,x, batchsize )
# %DBNCLASSPROBS calculates p(y|x) in a classification DBN
# %
# %  INPUTS
# %   dbn       : A dbn struct
# %   x         : matrix of samples  (n_samlples-by-n_features)
# %   batchsize : optionally takes a minibatch size in which case the result
# %               is calculated in minibatches to save memory
# %
# %  OUTPUT
# %   class_probs : class probabilites for each class (n_samples-by-n_classes)
# %
# %  EXAMPLE
# %   class_probs = dbnclassprobs( dbn,x )
# %   pred        = predict(x)
# %
# % See also, DBNPREDICT
#
# % Copyright Sřren Sřnderby july 2014


def dbnclassprobs(dbn, x, batchsize=None):

    n_rbm = len(dbn) - 1  # o: n_rbm = len(dbn.rbm) \ o: n_rbm = len(dbn)
    if not dbn[n_rbm].classRBM:
        raise ValueError("Class probabilities can only be calc. for classification DBN")

    # pass data deterministicly from input to top RBM
    for i in range(n_rbm-1):
        x = dbn[i].rbmup(dbn[i], x, [], sigm)

    batchsize = 1

    # at top RBM calculate class probabilities
    if batchsize or 'var':
        class_probs = rbmclassprobs(dbn[n_rbm], x, batchsize)
    else:
        class_probs = rbmclassprobs(dbn[n_rbm], x, batchsize)

    return class_probs


