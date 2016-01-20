import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_weighthed_hists(train):

    for col in train.columns:
        plt.clf()
        try:
            weights = np.ones_like(train[col].values)/float(len(train[col]))
            plt.hist(train[col].values, bins=51, alpha = 0.4, histtype='step', weights=weights)
            weights = np.ones_like(test[col].values)/float(len(test[col]))
            plt.hist(test[col].values, bins=51, alpha = 0.4, histtype='step', weights=weights)
            plt.savefig('plots/hist-%s.png' % col)
        except:
            print('bad %s' % col)


def count_outliers(train, categorical_columns=[]):
    for var in (train.columns - categorical_columns):
        high, low = train[var].mean() + 2* train[var].std(), train[var].mean() - 2* train[var].std()
        print("variable: %s" %var)
        highcount = np.sum(train[var] > high)
        lowcount = np.sum(train[var] < low)

        highmean = np.mean(y[np.where(train[var] > high)])
        lowmean = np.mean(y[np.where(train[var] < low)])

        print("upper outliers: mean: %f, count: %d" % (highmean, highcount))
        print("lower outliers: mean: %f, count: %d" % (lowmean, lowcount))


def plot_unq_values_meanvals(train, y):
    for var in train.columns:
        uniq = pd.unique(train[var])
        if len(uniq) < 200:
            meanz = [y[np.where(train[var] == i)].mean() for i in uniq]
            plt.clf()
            plt.plot(uniq, meanz, 'o')
            plt.savefig('plots/meanz-%s.png' % var)
        

def plot_meanvals(train, y):
    data = train.copy()
    data['target'] = np.array(y)
    for var in train.columns:
        dat = data[[var, 'target']]
        if len(pd.unique(dat[var])) > 100:
            # binning of continuous variable
            dat[var] = pd.cut(dat[var], 100, labels=False).values
        amean = dat.groupby([var],as_index=False).mean()
        astd = dat.groupby([var],as_index=False).std()

        plt.clf()
        plt.errorbar(amean[var], amean.target, 
                     yerr=astd.target, fmt='o')
        plt.ylim((0,1))
        plt.savefig('plots/means-%s.png' % var)

def plot_hists_compare01(train, y):
 
    for var in train.columns:
        plt.clf()
        unq = pd.unique(trainclip[var])
        if len(unq) < 40:
            sub0, sub1 = train[var].hist(bins=len(unq), by=y)
        else:
            sub0, sub1 = train[var].hist(bins=40, by=y)
        ymax = np.max([sub0.get_ylim() ,sub1.get_ylim()])
        sub0.set_ylim(top=ymax)
        sub1.set_ylim(top=ymax) 
        plt.savefig('histm-%s.png' % var)
        plt.close()
