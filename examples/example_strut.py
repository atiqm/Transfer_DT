import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../Class_Imb_Strut/')
import lib_tree
import STRUT

# =============================================================================
#
# =============================================================================

import numpy as np
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_I():
    from sklearn.datasets import load_iris
    iris = load_iris()

    inds = np.where(iris.data[:, 3] > np.median(iris.data[:, 3]))[0]
    indt = np.where(iris.data[:, 3] <= np.median(iris.data[:, 3]))[0]

    X_source = iris.data[np.concatenate((inds, indt[:5]))]
    y_source = iris.target[np.concatenate((inds, indt[:5]))]

    X_target_005 = iris.data[np.concatenate((inds[-10:], indt[:5]))][::2]
    y_target_005 = iris.target[np.concatenate((inds[-10:], indt[:5]))][::2]

    X_target_095 = iris.data[np.concatenate((inds[-10:], indt[:5]))][1::2]
    y_target_095 = iris.target[np.concatenate((inds[-10:], indt[:5]))][1::2]
    return [X_source, X_target_005,
            X_target_095, y_source,
            y_target_005, y_target_095]


def load_6():
    from sklearn.datasets import load_digits
    digits = load_digits()

    X = digits.data[:200]
    y = (digits.target[:200] == 6).astype(int)

    X_targ = digits.data[200:]
    y_targ = (digits.target[200:] == 9).astype(int)

    X_source = X
    y_source = y

    # separating 5% & 95% of target data, stratified, random
    X_target_095, X_target_005, y_target_095, y_target_005 = train_test_split(
        X_targ,
        y_targ,
        test_size=0.05,
        stratify=y_targ)

    return [X_source, X_target_005,
            X_target_095, y_source,
            y_target_005, y_target_095]

# =============================================================================
#
# =============================================================================

print('EXAMPLE STRUT')

#X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_I()
X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_6()

MAX = 5
solo_tree = True


def true_pos(clf, X, y):
    return sum(clf.predict(X[y == 1]) == 1) / sum(y == 1)


def false_pos(clf, X, y):
    return sum(clf.predict(X[y == 0]) == 1) / sum(y == 0)

if solo_tree:

    # =============================================================================
    #     ON A UNIQUE DECISION TREE
    # =============================================================================

    dtree_or = DecisionTreeClassifier(max_depth=MAX)
    dtree_or.fit(X_source, y_source)

    cl_no_red = [1]
    Nkmin = sum(y_target_005 == cl_no_red)
    root_source_values = lib_tree.get_node_distribution(dtree_or, 0).reshape(-1)

    props_s = root_source_values
    props_s = props_s / sum(props_s)
    props_t = np.zeros(props_s.size)
    for k in range(props_s.size):
        props_t[k] = np.sum(y_target_005 == k) / y_target_005.size

    coeffs = np.divide(props_t, props_s)

    strut = copy.deepcopy(dtree_or)
    strut_no_div = copy.deepcopy(dtree_or)
    strut_imb = copy.deepcopy(dtree_or)

    STRUT.STRUT(strut, 0, X_target_005, y_target_005)
    STRUT.STRUT(strut_no_div, 0, X_target_005, y_target_005, use_divergence=False)
    STRUT.STRUT(strut_imb, 0, X_target_005, y_target_005, adapt_prop=True, coeffs=coeffs)

    print('score strut:', strut.score(X_target_095, y_target_095))
    print('score strut no div:', strut_no_div.score(X_target_095, y_target_095))
    print('score strut*:', strut_imb.score(X_target_095, y_target_095))

    print('tpr strut:', true_pos(strut, X_target_095, y_target_095))
    print('tpr strut no div:', true_pos(strut_no_div, X_target_095, y_target_095))
    print('tpr strut*:', true_pos(strut_imb, X_target_095, y_target_095))

    print('fpr strut:', false_pos(strut, X_target_095, y_target_095))
    print('fpr strut no div:', false_pos(strut_no_div, X_target_095, y_target_095))
    print('fpr strut*:', false_pos(strut_imb, X_target_095, y_target_095))

#    print('nb feuilles strut :',sum(strut.tree_.feature == -2))
#    print('nb feuilles strut :',sum(strut_no_div.tree_.feature == -2))
#    print('nb feuilles strut*:',sum(strut_imb.tree_.feature == -2))

else:

    # =============================================================================
    #     ON A RANDOM FOREST
    # =============================================================================

    N_EST = 3

    rf_or = RandomForestClassifier(n_estimators=N_EST, max_depth=MAX)
    rf_or.fit(X_source, y_source)

    rf_strut = STRUT.STRUT_RF(rf_or, X_target_005, y_target_005)
    rf_strut_no_div = STRUT.STRUT_RF(rf_or, X_target_005, y_target_005, use_divergence=False)
    rf_strut_imb = STRUT.STRUT_RF(rf_or, X_target_005, y_target_005, adapt_prop=True)

    print('score strut:', rf_strut.score(X_target_095, y_target_095))
    print('score strut no div:', rf_strut_no_div.score(X_target_095, y_target_095))
    print('score strut*:', rf_strut_imb.score(X_target_095, y_target_095))

    print('tpr strut:', true_pos(rf_strut, X_target_095, y_target_095))
    print('tpr strut no div:', true_pos(rf_strut_no_div, X_target_095, y_target_095))
    print('tpr strut*:', true_pos(rf_strut_imb, X_target_095, y_target_095))

    print('fpr strut:', false_pos(rf_strut, X_target_095, y_target_095))
    print('fpr strut no div:', false_pos(rf_strut_no_div, X_target_095, y_target_095))
    print('fpr strut*:', false_pos(rf_strut_imb, X_target_095, y_target_095))
