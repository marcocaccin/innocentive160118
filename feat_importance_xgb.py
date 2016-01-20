import numpy as np
import pandas as pd
import xgboost as xgb
import operator

train = pd.read_csv('train-yes.csv')
y = np.loadtxt('y-yes.csv', dtype='int')

params = {
        "max_depth"             : 8,
        "eta"                   : 0.005,
        "early_stopping_rounds" : 100,
        "objective"             : 'binary:logistic',
        "subsample"             : 0.8,
        "lambda"                : 2,
        "colsample_bytree"      : 1.,
        "n_jobs"                : -1,
        "n_estimators"          : 1000,
        "silent"                : 1,
        "min_child_weight"      : 1,
        "eval_metric"           : 'auc'}

dtrain = xgb.DMatrix(np.array(train), label=y)
columns = list(train.columns)

train = None

model = xgb.train(params,
                  dtrain,
                  params['n_estimators'],
                  [(dtrain, 'train')])

importance = model.get_fscore()

importance_feature = {}
for k,w in importance.iteritems():
    number = int(k.strip('f'))
    importance_feature[columns[number]] = w

importance = sorted(importance_feature.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.to_csv('feat_importance_xgb.csv')
