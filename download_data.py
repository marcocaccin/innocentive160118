from urllib import urlretrieve
import pandas as pd
import os
from zipfile import ZipFile
import numpy as np
from sklearn.preprocessing import LabelEncoder

baseurl = 'https://s3.amazonaws.com/9933493/'
basename = 'InnoCentive_Challenge_9933493_'

if not os.path.exists('data'):
    os.mkdir('data')

total = []
for dataname in ['train', 'test']:
    urlretrieve('%s%s%sing_data.zip' % (baseurl, basename, dataname), 
                'data/%s.zip' % dataname)
    os.system('unzip data/%s.zip' % dataname)
    df = pd.read_csv('%s%sing_data.csv' % (basename, dataname))
    os.remove('%s%sing_data.csv' % (basename, dataname))
    os.remove('data/%s.zip' % dataname)

    if dataname == 'train':
        y = df['target_purchase'].values
        np.savetxt('data/y_train.csv', y, fmt='%d')

    df.drop(['target_purchase', 'record_ID'] , inplace=True, axis=1)
    df.fillna(-1)
    total.append(df)

ntrain = len(total[0])
total = pd.concat(total)
cats = []
print("Label encoding...")
for feat in total.columns:
    if total[feat].dtype=='object':
        cats.append(feat)
        print('Converting %s...' % feat)
        le = LabelEncoder()
        total[feat] = le.fit_transform(list(total[feat].values))
np.savetxt('categorical_features.csv', cats, fmt='%s')

train = total[:ntrain]
test = total[ntrain:]
total = None
print("Saving data frames...")
train.to_csv('data/train-yes.csv', index=False)
test.to_csv('data/test-yes.csv', index=False)
