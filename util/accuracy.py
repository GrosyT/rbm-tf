import sys
sys.path.insert(0, './util/')
from predict import predict
import numpy as np

# # ACCURACY calculates accuracy
#
#  INPUTS
#     pred_probs : predicted probabilities for each class
#             ey : one-of-K encoded true classes
#
#   OUTPUT
#            err : accuracy error. The output from error function must be some
#                  error emasure.
# other_measures : optionally output a struct with other error measures. These
#                  are not used but stored in opts.val_error_measures and
#                  opts.train_error_measures.
#
#  Copyright (c) Sřren Sřnderby july 2014

# find predictions and correct labels

# note that accuracy is called with normalized values


def accuracy(opts, pred_probs, ey):
    pred = predict(pred_probs)
    expected = predict(ey)  # because normalized x

    other_measures = {}
    err = 1 - np.mean(pred == expected)

    return err, other_measures
