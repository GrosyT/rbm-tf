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

n_rbm = len(dbn.rbm)

