import sys
sys.path.insert(0, './util/')
from predict import predict
from dbnclassprobs import dbnclassprobs
# DBNPREDICT predict labels using classification DBN
#
#  INPUTS
#   dbn : A dbn struct
#   x   : matrix of samples  (n_samlples-by-n_features)
#
#  OUTPUT
#   predictions : [n_samples x 1] vector of predicted labels
#
# See also, DBNCLASSPROBS
#
# Copyright Sřren Sřnderby July 2014


def dbnpredict(dbn, x):
    class_probs = dbnclassprobs(dbn, x)
    predictions = predict(class_probs)

    return predictions