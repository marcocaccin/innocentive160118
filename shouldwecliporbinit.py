from __future__ import print_function, division
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing, linear_model, metrics, cross_validation
from scipy import stats
from matplotlib import pyplot as plt
import seaborn

train = pd.read_csv('train-yes.csv')
y= np.load('y_train.npy')
clf = xgb.XGBClassifier(n_estimators=50,
                        nthread=-1,
                        max_depth=12,
                        learning_rate=0.05,
                        silent=True,
                        subsample=0.8,
                        colsample_bytree=0.7)

trainbin = train.copy()
for var in trainbin.columns:
    if len(trainbin[var].unique()) > 40:
        width = trainbin[var].ptp()
        minx  = trainbin[var].min()
        binned = pd.cut(train[var], bins = 40, right=False, labels=False)
        binned = binned * width / binned.ptp() + minx
        trainbin[var] = np.array(binned)

scoreclip = np.mean(cross_validation.cross_val_score(clf, train.values[:100000], y[:100000], cv=4)[0])

clipyes = []
# is it good to clip the variable?
train0 = pd.read_csv('train-yes.csv')
cat_vars = np.loadtxt('cats.csv', dtype='str')
train0.drop(cat_vars, inplace=True, axis=1)
score0 = np.mean(cross_validation.cross_val_score(clf, train0.values, y, cv=4)[0])
for var in train.columns:
    if len(train[var].unique()) >= 40:
        traino = train0.copy()
        traino[var] = train[var].values.copy()
        score = np.mean(cross_validation.cross_val_score(clf, traino, y, cv=4)[0])
        print("%s: %f" % (var, score - score0))
        if score - score0 > 0.005:
            clipyes.append(var)
"""
C10: 0.000240
C12: -0.000720
C18: 0.000000
C19: 0.000000
C27: 0.000000
C58: 0.000000
C59: 0.000000
C61: 0.000080
C62: 0.001720
C63: 0.001760
C64: 0.001120
C65: 0.000840
C66: 0.000960
C67: 0.000800
C68: 0.000200
C69: -0.000280
C70: 0.001160
C71: 0.000880
C72: 0.000960
C73: -0.000640
"""
# good2clip = ['C62', 'C63', 'C64', 'C65', 'C66', 'C67', 'C70', 'C71', 'C72']


# is it good to bin the float variable?
for var in trainbin.columns:
    if len(train[var].unique()) > 40:
        traino = train[:100000].copy()
        traino[var] = trainbin[var].values[:100000]
        scoreclipbin = np.mean(cross_validation.cross_val_score(clf, traino.values, y[:100000], cv=4)[0])
        print('%s: %f' % (var, scoreclipbin - scoreclip))

"""
C10: -0.001080
C12: -0.001680
C18: -0.000280
C19: 0.002800
C27: -0.001640
C58: 0.000640
C59: -0.000440
C61: 0.000000
C62: 0.001160
C63: -0.000280
C64: -0.000240
C65: -0.000240
C66: 0.001800
C67: 0.000920
C68: -0.000200
C69: -0.001520
C70: -0.000920
C71: -0.000480
C72: -0.000520
C73: 0.000040

C10: 0.724451
C12: 0.724251
C18: 0.723451
C19: 0.724411
C23: 0.724651
C24: 0.724251
C25: 0.724091
C26: 0.724411
C27: 0.723731
C58: 0.723611
C59: 0.724491
C61: 0.723971
C62: 0.724171
C63: 0.724611
C64: 0.725531
C65: 0.723571
C66: 0.724171
C67: 0.725451
C68: 0.723091
C69: 0.725691
C70: 0.723731
C71: 0.724411
C72: 0.725051
C73: 0.725811
"""
