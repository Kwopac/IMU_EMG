from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
RANDOM_STATE = 8
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score 
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

#Importing and Splitting the dataset 
#dataset = pd.read_csv('sfd.csv') 
dataset = pd.read_csv('5TAPFM_Curlup_features.csv') 
#df = dataset.iloc[ : , :] 
X, y = dataset.iloc[:, :10], dataset.iloc[:, -1]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=RANDOM_STATE)

for model in [LogisticRegression(random_state=RANDOM_STATE), RandomForestClassifier(random_state=RANDOM_STATE), MLPClassifier(max_iter=10000,random_state=RANDOM_STATE), XGBClassifier(),GradientBoostingClassifier(random_state=RANDOM_STATE), svm.SVC(), DecisionTreeClassifier(random_state=RANDOM_STATE),BaggingClassifier(n_estimators=50), AdaBoostClassifier(n_estimators=100)]:
    print("[INFO]: Fitting", str(model), "...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test , y_pred),"\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))

model = AdaBoostClassifier(n_estimators=100)
model.fit(X_train.values, y_train.values)
y_pred = model.predict(X_test.values)
print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred))

model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train.values, y_train.values)
y_pred = model.predict(X_test.values)
print("RF Model Accuracy:", accuracy_score(y_test, y_pred))

import pickle
pickle.dump(model, open('model2_picklefile.pkl','wb'))
model = pickle.load(open('model2_picklefile.pkl','rb'))



import glob
import nolds
import statistics
import numpy as np
import pandas as pd
import antropy as ant
from scipy.stats import entropy
from scipy import signal
from scipy.stats import linregress
from scipy.stats import kurtosis
import scipy.signal


path = r'C:\Users\priya\Desktop\Testdataset-m3'
csv_files = glob.glob(path + "/*.csv")


def ent(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

def rms(x):
    return np.sqrt(np.mean(x**2))

def var(x):
    return statistics.variance(x)

def meanfreq(x):
    fs = 6.25
    f, Pxx_den = signal.periodogram(x, fs)                                                    
    Pxx_den = np.reshape( Pxx_den, (1,-1) ) 
    width = np.tile(f[1]-f[0], (1, Pxx_den.shape[1]))
    f = np.reshape(f, (1, -1))
    P = Pxx_den * width
    pwr = np.sum(P)
    mnfreq = np.dot(P, f.T)/pwr
    return mnfreq[0][0]

def psd(x):
    (f, S) = scipy.signal.periodogram(x, 10, scaling='density')
    return linregress(f,S).slope

def dfa(x):
    return ant.detrended_fluctuation(x)

def lya(x):
    return nolds.lyap_r(x)*100

def kur(x):
    return kurtosis(x)

def meanabsdev(x):
    return x.mad()

def removeShimmer(string):
    return string[13:]

testpredict = []

for file in csv_files:
    d = pd.read_csv(file)
    datare = d.rename(columns = removeShimmer)

    lnx = datare.loc[:,'Accel_LN_X_CAL']
    lny = datare.loc[:,'Accel_LN_Y_CAL']
    lnz = datare.loc[:,'Accel_LN_Z_CAL']
    wrx = datare.loc[:,'Accel_WR_X_CAL']
    wry = datare.loc[:,'Accel_WR_Y_CAL']
    wrz = datare.loc[:,'Accel_WR_Z_CAL']
    mx = datare.loc[:,'Mag_X_CAL']
    my = datare.loc[:,'Mag_Y_CAL']
    mz = datare.loc[:,'Mag_Z_CAL']
    gx = datare.loc[:,'Gyro_X_CAL']
    gy = datare.loc[:,'Gyro_Y_CAL']
    gz = datare.loc[:,'Gyro_Z_CAL']
    
    #MODEL-1
    # PSD_gy = psd(gy)
    # Lyapunov_exponent_wrz = lya(wrz)
    # Variance_gy    = var(gy)
    # MAD_gy = meanabsdev(gy)
    # RMS_gy = rms(gy)
    # Entropy_wrx = ent(wrx)    
    # DFA_gy = dfa(gy)    
    # Variance_wrx = var(wrx)
    # Variance_lnx = var(lnx)
    # DFA_lnx = dfa(lnx)
    #features = [PSD_gy, Lyapunov_exponent_wrz, Variance_gy, MAD_gy, RMS_gy, Entropy_wrx, DFA_gy, Variance_wrx, Variance_lnx, DFA_lnx][
    
    #MODEL-2
    # Entropy_wrz = ent(wrz)    
    # Kurtosis_lnx = kur(lnx)
    # Lyapunov_exponent_lnz = lya(lnz)    
    # Kurtosis_wrx = kur(wrx)    
    # Kurtosis_wrz = kur(wrz)    
    # Kurtosis_mz = kur(mz)    
    # Kurtosis_lnz = kur(lnz)    
    # Variance_gy = var(gy)    
    # RMS_wry = rms(wry)    
    # Lyapunov_exponent_my = lya(my)
    # features = [Entropy_wrz, Kurtosis_lnx,    Lyapunov_exponent_lnz,    Kurtosis_wrx    ,Kurtosis_wrz    ,Kurtosis_mz,    Kurtosis_lnz,    Variance_gy,    RMS_wry    ,Lyapunov_exponent_my]
    
    #MODEL-3
    DFA_my = dfa(my)
    Kurtosis_lny = kur(lny)
    Mean_frequency_gz = meanfreq(gz)    
    Entropy_my = ent(my)
    Entropy_wry = ent(wry)
    PSD_wrx = psd(wrx)    
    Variance_wrx = var(wrx)    
    Mean_frequency_gy = meanfreq(gy)
    Lyapunov_exponent_lny = lya(lny)
    Kurtosis_gx = kur(gx)
    features = [DFA_my,	Kurtosis_lny	,Mean_frequency_gz	,Entropy_my,	Entropy_wry	,PSD_wrx	,Variance_wrx,	Mean_frequency_gy,	Lyapunov_exponent_lny,	Kurtosis_gx]
        
    
    #MODEL-4
    # Entropy_wry    = ent(wry)
    # Lyapunov_exponent_wrz = lya(wrz)
    # Kurtosis_my = kur(my)
    # Lyapunov_exponent_lnz = lya(lnz)
    # Entropy_lny = ent(lny)
    # Entropy_gz = ent(gz)    
    # Entropy_lnz    = ent(lnz)
    # DFA_wry = dfa(wry)
    # Kurtosis_wrx = kur(wrx)    
    # Lyapunov_exponent_lnx = lya(lnx)
    # features = [Entropy_wry,    Lyapunov_exponent_wrz,    Kurtosis_my,    Lyapunov_exponent_lnz    ,Entropy_lny,    Entropy_gz,    Entropy_lnz,    DFA_wry,    Kurtosis_wrx,    Lyapunov_exponent_lnx]
    
    #MODEL-5
    # Kurtosis_lnx = kur(lnx)    
    # Lyapunov_exponent_lny = lya(lny)
    # Kurtosis_gz = kur(gz)        
    # Kurtosis_wrx = kur(wrx)        
    # Variance_gy = var(gy) 
    # Entropy_gz = ent(gz)    
    # Lyapunov_exponent_lnx = lya(lnx)
    # Kurtosis_mx = kur(mx)        
    # Mean_frequency_my = meanfreq(my)     
    # Entropy_lny = ent(lny)     
    # features = [Kurtosis_lnx,Lyapunov_exponent_lny,    Kurtosis_gz,    Kurtosis_wrx,    Variance_gy,    Entropy_gz    ,Lyapunov_exponent_lnx    ,Kurtosis_mx    ,Mean_frequency_my,    Entropy_lny]

    
    #MODEL-7
    # Lyapunov_exponent_wrx = lya(wrx)
    # Entropy_wrz = ent(lnz)
    # Entropy_wry    = ent(wry)
    # DFA_gy = dfa(gy)
    # Lyapunov_exponent_lnz = lya(lnz)    
    # Mean_frequency_gy = meanfreq(gy)    
    # PSD_wrx    = psd(wrx)
    # Mean_frequency_lnx = meanfreq(lnx)    
    # Entropy_lnz    = ent(lnz)
    # Lyapunov_exponent_wry = lya(wry)
    # features = [Lyapunov_exponent_wrx,    Entropy_wrz,    Entropy_wry    ,DFA_gy,    Lyapunov_exponent_lnz,    Mean_frequency_gy,    PSD_wrx    ,Mean_frequency_lnx    ,Entropy_lnz,    Lyapunov_exponent_wry]

    #MODEL-6
    # Mean_frequency_my = meanfreq(my)    
    # RMS_lny = rms(lny)    
    # Kurtosis_gy = kur(gy)    
    # Entropy_gz = ent(gz)    
    # Entropy_wrz = ent(wrz)    
    # Mean_frequency_wry = meanfreq(wry)
    # Entropy_gx = ent(gx)    
    # PSD_gz = psd(gz)    
    # Variance_lnz = var(lnz)
    # DFA_wrx = dfa(wrx)
    # features = [Mean_frequency_my,    RMS_lny    ,Kurtosis_gy    ,Entropy_gz,    Entropy_wrz,    Mean_frequency_wry,    Entropy_gx,    PSD_gz,    Variance_lnz    ,DFA_wrx]
    
    final_features = np.array(features).reshape(1,10) 
    #print(model.predict(final_features))
    
    
    if model.predict(final_features)==1:
        testpredict.append(1)
    if model.predict(final_features)==0:
        testpredict.append(0)


predictres = []; Incorrectprediction = 0
origdata = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1]

for i in range(24):
    if testpredict[i] == origdata[i]:
        predictres.append(u'\u2713')
    else: 
        predictres.append('X')
        Incorrectprediction += 1
        
print(Incorrectprediction)  
        




