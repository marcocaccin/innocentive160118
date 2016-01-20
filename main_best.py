import pandas as pd
import numpy as np
import xgboost as xgb


########################################
########################################



    ##### Load data ######

    train = pd.read_csv('train-yes.csv')
    test = pd.read_csv('test-yes.csv')
    y = np.loadtxt('y-yes.csv').astype('int')
    train['C3'] = np.loadtxt('train-C3-mapped.csv')
    test['C3'] = np.loadtxt('test-C3-mapped.csv')

    ##### Convert categorical variables into binary (one hot encoder) ######
    total = pd.concat([train, test])
    ntrain = len(train)
    for feat in categorical_features:
        dum = pd.get_dummies(total[feat], prefix=feat)
        total.drop(feat, axis=1, inplace=True)
        for binfeat in dum.columns:
            total[binfeat] = dum[binfeat].values.copy()
    train = total[:ntrain]; test  = total[ntrain:]; total = None

    ##### Train a model for prediction  ######

    params = {
        "max_depth"             : 13,
        "eta"                   : 0.005,
        "early_stopping_rounds" : 100,
        "objective"             : 'binary:logistic',
        "subsample"             : 0.8,
        "lambda"                : 1.,
        "colsample_bytree"      : 0.8,
        "n_jobs"                : -1,
        "n_estimators"          : 3300,
        "silent"                : 1,
        "min_child_weight"      : 10,
        "eval_metric"           : 'error'}

    dtrain = xgb.DMatrix(np.array(train), label=y)
    dtest  = xgb.DMatrix(np.array(test))
    train = None
    test = None
    
    print("Teach...")
    model = xgb.train(params,
                      dtrain,
                      params['n_estimators'])
    
    preds = model.predict(dtest)
    predz = (preds > 0.5).astype('int')
    np.savetxt('pred_mapd_dumm.csv', predz, fmt='%d', header='target_purchase', comments='')
