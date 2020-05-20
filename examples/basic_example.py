import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
sys.path.insert(0, '../')
import ser
import strut

methods = [
    'ser',
    'strut',
    'ser_nr',
    'strut_hi'
]
labels = [
    'SER',
    'STRUT',
    'SER$^{*}$',
    # 'STRUT$^{*}$',
    'STRUT$^{*}$',
]

np.random.seed(0)

plot_step = 0.01
# Generate training source data
ns = 200
ns_perclass = ns // 2
mean_1 = (1, 1)
var_1 = np.diag([1, 1])
mean_2 = (3, 3)
var_2 = np.diag([2, 2])
Xs = np.r_[np.random.multivariate_normal(mean_1, var_1, size=ns_perclass),
           np.random.multivariate_normal(mean_2, var_2, size=ns_perclass)]
ys = np.zeros(ns)
ys[ns_perclass:] = 1
# Generate training target data
nt = 50
# imbalanced
nt_0 = nt // 10
mean_1 = (6, 3)
var_1 = np.diag([4, 1])
mean_2 = (5, 5)
var_2 = np.diag([1, 3])
Xt = np.r_[np.random.multivariate_normal(mean_1, var_1, size=nt_0),
           np.random.multivariate_normal(mean_2, var_2, size=nt - nt_0)]
yt = np.zeros(nt)
yt[nt_0:] = 1
# Generate testing target data
nt_test = 1000
nt_test_perclass = nt_test // 2
Xt_test = np.r_[np.random.multivariate_normal(mean_1, var_1, size=nt_test_perclass),
                np.random.multivariate_normal(mean_2, var_2, size=nt_test_perclass)]
yt_test = np.zeros(nt_test)
yt_test[nt_test_perclass:] = 1

# Source classifier
clf_source = DecisionTreeClassifier(max_depth=None)
clf_source.fit(Xs, ys)
score_src_src = clf_source.score(Xs, ys)
score_src_trgt = clf_source.score(Xt_test, yt_test)
print('Training score Source model: {:.3f}'.format(score_src_src))
print('Testing score Source model: {:.3f}'.format(score_src_trgt))
clfs = []
scores = []
# Transfer with SER
for method in methods:
    clf_transfer = copy.deepcopy(clf_source)
    if method == 'ser':
        ser.SER(0, clf_transfer, Xt, yt, original_ser=True)
    if method == 'ser_nr':
        ser.SER(0, clf_transfer, Xt, yt,
                original_ser=False,
                no_red_on_cl=True,
                cl_no_red=[0],
                ext_cond=True)
    if method == 'strut':
        strut.STRUT(clf_transfer, 0, Xt, yt, Xt, yt)
    if method == 'strut_hi':
        strut.STRUT(clf_transfer, 0, Xt, yt, Xt, yt,
                    pruning_updated_node=True,
                    no_prune_on_cl=False,
                    adapt_prop=True,
                    simple_weights=False,
                    coeffs=[0.2, 1])
    score = clf_transfer.score(Xt_test, yt_test)
    print('Testing score transferred model ({}) : {:.3f}'.format(method, score))
    clfs.append(clf_transfer)
    scores.append(score)

# Plot decision functions

# Data on which to plot source
x_min, x_max = Xs[:, 0].min() - 1, Xs[:, 0].max() + 1
y_min, y_max = Xs[:, 1].min() - 1, Xs[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
# Plot source model
Z = clf_source.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
fig, ax = plt.subplots(nrows=1, ncols=len(methods) + 1, figsize=(13, 3))
ax[0].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
ax[0].scatter(Xs[0, 0], Xs[0, 1],
              marker='o',
              edgecolor='black',
              color='white',
              label='source data',
              )
ax[0].scatter(Xs[:ns_perclass, 0], Xs[:ns_perclass, 1],
              marker='o',
              edgecolor='black',
              color='blue',
              )
ax[0].scatter(Xs[ns_perclass:, 0], Xs[ns_perclass:, 1],
              marker='o',
              edgecolor='black',
              color='red',
              )
ax[0].set_title('Model: Source\nAcc on source data: {:.2f}\nAcc on target data: {:.2f}'.format(score_src_src, score_src_trgt),
                fontsize=11)
ax[0].legend()

# Data on which to plot target
x_min, x_max = Xt[:, 0].min() - 1, Xt[:, 0].max() + 1
y_min, y_max = Xt[:, 1].min() - 1, Xt[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
# Plot transfer models
for i, (method, label, score) in enumerate(zip(methods, labels, scores)):
    clf_transfer = clfs[i]
    Z_transfer = clf_transfer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_transfer = Z_transfer.reshape(xx.shape)
    ax[i + 1].contourf(xx, yy, Z_transfer, cmap=plt.cm.coolwarm, alpha=0.8)
    ax[i + 1].scatter(Xt[0, 0], Xt[0, 1],
                      marker='o',
                      edgecolor='black',
                      color='white',
                      label='target data',
                      )
    ax[i + 1].scatter(Xt[:nt_0, 0], Xt[:nt_0, 1],
                      marker='o',
                      edgecolor='black',
                      color='blue',
                      )
    ax[i + 1].scatter(Xt[nt_0:, 0], Xt[nt_0:, 1],
                      marker='o',
                      edgecolor='black',
                      color='red',
                      )
    ax[i + 1].set_title('Model: {}\nAcc on target data: {:.2f}'.format(label, score),
                        fontsize=11)
    ax[i + 1].legend()

# fig.savefig('../images/ser_strut.png')
plt.show()
