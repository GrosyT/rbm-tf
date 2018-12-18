import numpy as np
# PREDICT find most likely class i.e idx of max value in each row


def predict(x):
    p = np.argmax(x, axis=1)
    return p