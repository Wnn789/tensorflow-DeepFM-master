import numpy as np


def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


if __name__ == '__main__':
    actual = [1,1,2,4,5,5]
    pred = [0.9,0.9,1.9,4.1,5.1,5.1]
    pred_1 = [5,3,2,4,1,2]
    print(gini_norm(actual,pred))
    print(gini_norm(actual,pred_1))