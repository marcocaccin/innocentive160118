import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import sklearn.cross_validation as cv
import itertools


def evalauc(model, X_teach, X_valid, y_teach, y_valid):
    model.fit(X_teach, y_teach)
    preds = model.predict(X_valid)
    return metrics.roc_auc_score(y_valid, preds)


print("Loading data...")
train = pd.read_csv('data/train-yes.csv')
y = np.loadtxt('data/y_train.csv', dtype='int')
cats = np.loadtxt('categorical_variables.csv', dtype='str')
train = train.drop(cats, axis=1)
train['target'] = y

# reduce data size to 50k
train = train.sample(n=50000)
y = train['target'].values.copy()
train = train.drop('target', axis=1)

X_teach, X_valid, y_teach, y_valid = cv.train_test_split(np.array(train), y,
                                                         test_size=0.3, random_state=51786)

columns = list(train.columns)
ncols = len(columns)
rfc = RandomForestClassifier(n_estimators=50)

benchmark_score = evalauc(rfc, X_teach, X_valid, y_teach, y_valid)
with open("bfeats.txt", "w") as myfile:
    myfile.write("base = %f" % benchmark_score)


better = []
istep = 0
for col0, col1 in itertools.combinations_with_replacement(np.arange(ncols), 2):

    istep +=1
    print(istep)
    
    X_teach2 = np.vstack((X_teach, X_teach[:,col0] + X_teach[:,col1]))
    X_valid2 = np.vstack((X_valid, X_valid[:,col0] + X_valid[:,col1]))
    score = evalauc(rfc, X_teach2, X_valid2, y_teach, y_valid)
    if score > benchmark_score:
        better.append("%s + %s" % (columns[0], columns[1]))
        print(better[-1])
        with open("bfeats.txt", "a") as myfile:
            myfile.write("%s+%s = %f" % (columns[0], columns[1], score))

    X_teach2 = np.vstack((X_teach, X_teach[:,col0] - X_teach[:,col1]))
    X_valid2 = np.vstack((X_valid, X_valid[:,col0] - X_valid[:,col1]))
    score = evalauc(rfc, X_teach2, X_valid2, y_teach, y_valid)
    if score > benchmark_score:
        better.append("%s-%s" % (columns[0], columns[1]))
        print(better[-1])
        with open("bfeats.txt", "a") as myfile:
            myfile.write("%s-%s = %f" % (columns[0], columns[1], score))

    X_teach2 = np.vstack((X_teach, X_teach[:,col0] * X_teach[:,col1]))
    X_valid2 = np.vstack((X_valid, X_valid[:,col0] * X_valid[:,col1]))
    score = evalauc(rfc, X_teach2, X_valid2, y_teach, y_valid)
    if score > benchmark_score:
        better.append("%s*%s" % (columns[0], columns[1]))
        print(better[-1])
        with open("bfeats.txt", "a") as myfile:
            myfile.write("%s*%s = %f" % (columns[0], columns[1], score))

    if 0 not in X_teach[:,col1]:
        X_teach2 = np.vstack((X_teach, X_teach[:,col0] / X_teach[:,col1]))
        X_valid2 = np.vstack((X_valid, X_valid[:,col0] / X_valid[:,col1]))
        score = evalauc(rfc, X_teach2, X_valid2, y_teach, y_valid)
        if score > benchmark_score:
            better.append("%s/%s" % (columns[0], columns[1]))
            print(better[-1])
            with open("bfeats.txt", "a") as myfile:
                myfile.write("%s/%s = %f" % (columns[0], columns[1], score))
