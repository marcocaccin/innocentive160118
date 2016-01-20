import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing, linear_model, metrics, decomposition
import sklearn.cross_validation as cv

##### Load data ######

train = pd.read_csv('train.csv').drop('record_ID', axis=1)
test = pd.read_csv('test.csv').drop('record_ID', axis=1)
y = train.target_purchase.values
train.drop('target_purchase', inplace=True, axis=1)
test.fillna(-1, inplace=True)

##### Convert C3 variable ######
C3 = train.C3.values.copy()
le = preprocessing.LabelEncoder()
C3t = le.fit_transform(np.atleast_2d(C3).T)
train.drop('C3', axis=1, inplace=True)

##### Convert categorical variables into numerical (label encode) ######
total = pd.concat([train, test])
categorical_features = ['C3']
for feat in total.columns:
    if total[feat].dtype=='object':
        total[feat] = le.fit_transform(list(total[feat].values))
        categorical_features.append(feat)
train = total[:ntrain]; test  = total[ntrain:]; total = None

##### Train a model for prediction of C3 given all other features ######
X_teach, X_valid, c3_teach, c3_valid = cv.train_test_split(np.array(train), C3t, test_size=0.3, random_state=0)

dtrain = xgb.DMatrix(X_teach, label=c3_teach); X_teach = None; c3_teach = None
dvalid = xgb.DMatrix(X_valid, label=c3_valid); X_valid = None; c3_valid = None

param = {
    'objective'       : 'multi:softmax',
    'eta'             : 0.05,
    'early_stopping_rounds' : 20,
    'silent'          : 1,
    'num_class'       : 12,
    'max_depth'       : 12,
    'colsample_bytree': 0.6,
    'nthread'         : 4,
    'subsample'       : 0.7}
watchlist = [ (dtrain,'train'), (dtest, 'test') ]

bst = xgb.train(param, dtrain, 200, watchlist)



##### Convert C3 in test set ######
subs = []
yprobs = []
conversions = []
grp = test.groupby('C3')
C3test = test['C3'].values.copy()
for val in pd.unique(test.C3.values):
    sub = grp.get_group(val)
    subs.append(sub)

    dtest = xgb.DMatrix(np.array(sub))
    yprob = bst.predict(dtest).astype('int')
    yprobs.append(yprob.copy())
    conversion = pd.value_counts(yprob).index[0]
    C3test[np.where(C3test == val)] = conversion

##### Put back C3 into features ######
train['C3'] = C3t
test['C3'] = C3test

np.savetxt('train-C3-mapped.csv', C3t)
np.savetxt('test-C3-mapped.csv', C3test)
