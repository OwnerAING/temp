#!/usr/bin/env python
# coding: utf-8


# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.io import loadmat


data = loadmat('hw03_dataset.mat')


X = data['X_trn']
y = data['Y_trn']
X_val = data['X_tst']
y_val = data['Y_tst']
X1 = X[:,0]
X2 = X[:,1]
X_val1 = X_val[:,0]
y = y.reshape(X1.shape)
y_val = y_val.reshape(X_val1.shape)


def plot_hyperplane(clf, X, y,
                    h=0.02,
                    draw_sv=True,
                    title='hyperplan'):

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)
    plt.scatter(X[:,0],X[:,1],c=y)


    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='r', marker='.',s=1)

    plt.show()



clf = svm.SVC(kernel='rbf',C=0.1,gamma=0.01)
clf.fit(X, y)
plot_hyperplane(clf, X, y, h=0.01, title="C={},gama{}".format(1,0.01))