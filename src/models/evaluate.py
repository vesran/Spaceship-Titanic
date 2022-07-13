import numpy as np


def get_confusion_matrix(y_true, y_pred):
    n = np.unique(y_true).shape[0]
    conf = np.zeros((n, n))
    for t, p in zip(y_true, y_pred):
        # 1-1 should be on the top left corner
        conf[n-t-1][n-p-1] += 1        
    # Normalize per truth
    conf = conf / conf.sum(axis=0)
    return np.round(conf, 4)
