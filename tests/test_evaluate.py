import numpy as np
from models.evaluate import get_confusion_matrix


def test_values():
    y_pred = [0, 1]
    y_true = [0, 1]
    conf = get_confusion_matrix(y_pred, y_true)
    assert conf[0][0] == 1
    assert conf[1][1] == 1
    assert conf[0][1] == 0
    assert conf[1][0] == 0