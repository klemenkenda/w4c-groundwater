import itertools

import numpy as np
from scipy.interpolate import splrep, splev
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


class Holder:
    def __init__(self, fst, snd):
        self.fst = fst
        self.snd = snd
        self.clip = lambda x: clip(x, self.fst, self.snd)


def clip(data, fst, snd):
    return np.append(data[:fst], data[snd:])


def all_options(n):
    combs = list(range(n))
    for k in range(1, n + 1):
        yield from itertools.combinations(combs, k)


def split_data(data, train_perc=0.7, seed=None):
    if seed is not None:
        np.random.seed(seed)
    num, size = data.shape
    print(num, size)
    inds = np.arange(0, size)
    # print(inds)
    np.random.shuffle(inds)
    train = data[:, inds[:int(train_perc * size)]]
    test = data[:, inds[int(train_perc * size):]]
    return train, test


def predict_all(inds, data, objective_ind, regressor, seed=2018):
    train, test = split_data(data, seed=seed)
    predictors = np.array(
        [inds[j] for j in range(len(inds)) if j != objective_ind])
    j = 0

    rtr = []
    for arr in all_options(len(predictors)):
        j += 1
        r2, fit = predict(train, test, list(arr), objective_ind, regressor(), j)
        assert len(fit.coef_) == len(list(arr))
        rtr.append((r2, list(arr), fit))

    rtr.sort(reverse=True)
    return rtr


def predict(train, test, predictors, objective, regressor, su):
    train_data = train[predictors]
    train_target = train[objective]
    assert objective not in predictors
    test_data = test[predictors]
    test_target = test[objective]

    fit = regressor.fit(train_data.T, train_target)

    r2 = r2_score(test_target, fit.predict(test_data.T))
    # print("r2 :","{0:1.4f} |".format(r2_score(test_target, fit.predict(test_data.T))))
    if su == 376:
        plt.plot(test_target)
        plt.plot(fit.predict(test_data.T))
        print("r2 :", "{0:1.4f} |".format(
            r2_score(test_target, fit.predict(test_data.T))))

    return r2, fit
