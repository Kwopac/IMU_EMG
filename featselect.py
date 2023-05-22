import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
from skfeature.function.similarity_based import fisher_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import mrmr
from sklearn import feature_selection as fs
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

dataset = pd.read_csv('model_3_features.csv') 
X, y = dataset.iloc[:, :108], dataset.iloc[:, -1]
head = X.columns
head = head.tolist()

#chi2
# chi2_features = SelectKBest(chi2, k = 11)
# X_kbest_features = chi2_features.fit_transform(X, y)

#mrmr
from mrmr import mrmr_classif
mrmrfeatures = mrmr_classif(X=X, y=y, K=30)

# feaaat = pd.DataFrame()
# for i in range(10):
#     xx = dataset.loc[:,mrmrfeatures[i]]
#     feaaat[mrmrfeatures[i]] = xx;
# feaaat['label'] = y
# feaaat.to_csv('feaaat.csv',mode='a',index = False)

#anova
fs_fit_fscore = fs.SelectKBest(fs.f_classif, k=30)
fs_fit_fscore.fit_transform(X, y)
fs_indices_fscore = np.argsort(np.nan_to_num(fs_fit_fscore.scores_))[::-1][0:30]
best_features_fscore = dataset.columns[fs_indices_fscore].values
fscorefeatures = best_features_fscore.tolist()

# feaaat = pd.DataFrame()
# for i in range(10):
#     xx = dataset.loc[:,fscorefeatures[i]]
#     feaaat[fscorefeatures[i]] = xx;
# feaaat['label'] = y
# feaaat.to_csv('feaaat.csv',mode='a',index = False)

#RFE
RFEfeatures = []
rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), n_features_to_select=30)
fit = rfe.fit(X, y)
names = dataset.columns.values[0:-1]
for i in range(len(fit.support_)):
    if fit.support_[i]:
        RFEfeatures.append(names[i])

# feaaat = pd.DataFrame()
# for i in range(10):
#     xx = dataset.loc[:,RFEfeatures[i]]
#     feaaat[RFEfeatures[i]] = xx;
# feaaat['label'] = y
# feaaat.to_csv('feaaat2.csv',mode='a',index = False)

#MutualInfo
def select_features(X_train, y_train, X_test):
  # configure to select all features
  fs = SelectKBest(score_func=mutual_info_classif, k='all')
  # learn relationship from training data
  fs.fit(X_train, y_train)
  # transform train input data
  X_train_fs = fs.transform(X_train)
  # transform test input data
  X_test_fs = fs.transform(X_test)
  return X_train_fs, X_test_fs, fs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

Dict = {}
for i in range(len(fs.scores_)):
  #print('Feature %d: %f' % (i, fs.scores_[i]))
  Dict[head[i]] =  fs.scores_[i]

sort_data = sorted(Dict.items(), key=lambda x: x[1], reverse=True)

sort_data_dict = dict(sort_data)
print(sort_data_dict)

    
mutualinfofeatures = []
for i in range(30):
    mutualinfofeatures.append(list(sort_data_dict)[i])
    
totalfeat = []
totalfeat.extend(mrmrfeatures)
totalfeat.extend(fscorefeatures)
totalfeat.extend(RFEfeatures)
totalfeat.extend(mutualinfofeatures)

from collections import Counter
aa = dict(Counter(totalfeat))

sort_data2 = sorted(aa.items(), key=lambda x: x[1], reverse=True)
sort_data_dict2 = dict(sort_data2)

rankedfeat = []
for i in range(10):
    rankedfeat.append(list(sort_data_dict2)[i])
    
Rankedfeatures = pd.DataFrame()

for i in range(10):
    xx = dataset.loc[:,list(sort_data_dict2)[i]]
    Rankedfeatures[list(sort_data_dict2)[i]] = xx;
    


Rankedfeatures['label'] = y
Rankedfeatures.to_csv('rankedproper_3.csv',mode='a',index = False)