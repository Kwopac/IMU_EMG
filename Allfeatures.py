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
        
path = r'C:\Users\priya\Desktop\3stu\7Reverse_Curlup'
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

Meanfrequency_lnx = []; Meanfrequency_lny = []; Meanfrequency_lnz = []
Meanfrequency_wrx = []; Meanfrequency_wry = []; Meanfrequency_wrz = []
Meanfrequency_mx = []; Meanfrequency_my = []; Meanfrequency_mz = []
Meanfrequency_gx = []; Meanfrequency_gy = []; Meanfrequency_gz = []

RMS_lnx = []; RMS_lny = []; RMS_lnz = []
RMS_wrx = []; RMS_wry = []; RMS_wrz = []
RMS_mx = []; RMS_my = []; RMS_mz = []
RMS_gx = []; RMS_gy = []; RMS_gz = []

kurtosis_lnx = []; kurtosis_lny = []; kurtosis_lnz = []
kurtosis_wrx = []; kurtosis_wry = []; kurtosis_wrz = []
kurtosis_mx = []; kurtosis_my = []; kurtosis_mz = []
kurtosis_gx = []; kurtosis_gy = []; kurtosis_gz = []

DFA_lnx = []; DFA_lny = []; DFA_lnz = []
DFA_wrx = []; DFA_wry = []; DFA_wrz = []
DFA_mx = []; DFA_my = []; DFA_mz = []
DFA_gx = []; DFA_gy = []; DFA_gz = []

Variance_lnx = []; Variance_lny = []; Variance_lnz = []
Variance_wrx = []; Variance_wry = []; Variance_wrz = []
Variance_mx = []; Variance_my = []; Variance_mz = []
Variance_gx = []; Variance_gy = []; Variance_gz = []

PSD_lnx = []; PSD_lny = []; PSD_lnz = []
PSD_wrx = []; PSD_wry = []; PSD_wrz = []
PSD_mx = []; PSD_my = []; PSD_mz = []
PSD_gx = []; PSD_gy = []; PSD_gz = []

Entropy_lnx = []; Entropy_lny = []; Entropy_lnz = []
Entropy_wrx = []; Entropy_wry = []; Entropy_wrz = []
Entropy_mx = []; Entropy_my = []; Entropy_mz = []
Entropy_gx = []; Entropy_gy = []; Entropy_gz = []

Lyapunov_lnx = []; Lyapunov_lny = []; Lyapunov_lnz = []
Lyapunov_wrx = []; Lyapunov_wry = []; Lyapunov_wrz = []
Lyapunov_mx = []; Lyapunov_my = []; Lyapunov_mz = []
Lyapunov_gx = []; Lyapunov_gy = []; Lyapunov_gz = []

mad_lnx = []; mad_lny = []; mad_lnz = []
mad_wrx = []; mad_wry = []; mad_wrz = []
mad_mx = []; mad_my = []; mad_mz = []
mad_gx = []; mad_gy = []; mad_gz = []

for file in csv_files:
    data = pd.read_csv(file)
    datare = data.rename(columns = removeShimmer)
    
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

    Meanfrequency_lnx.append(meanfreq(lnx)); Meanfrequency_lny.append(meanfreq(lny)); Meanfrequency_lnz.append(meanfreq(lnz))
    Meanfrequency_wrx.append(meanfreq(wrx)); Meanfrequency_wry.append(meanfreq(wry)); Meanfrequency_wrz.append(meanfreq(wrz))
    Meanfrequency_mx.append(meanfreq(mx)); Meanfrequency_my.append(meanfreq(my)); Meanfrequency_mz.append(meanfreq(mz))
    Meanfrequency_gx.append(meanfreq(gx)); Meanfrequency_gy.append(meanfreq(gy)); Meanfrequency_gz.append(meanfreq(gz))
    
    RMS_lnx.append(rms(lnx)); RMS_lny.append(rms(lny)); RMS_lnz.append(rms(lnz))
    RMS_wrx.append(rms(wrx)); RMS_wry.append(rms(wry)); RMS_wrz.append(rms(wrz))
    RMS_mx.append(rms(mx)); RMS_my.append(rms(my)); RMS_mz.append(rms(mz));
    RMS_gx.append(rms(gx)); RMS_gy.append(rms(gy)); RMS_gz.append(rms(gz));
    
    kurtosis_lnx.append(kur(lnx)); kurtosis_lny.append(kur(lny)); kurtosis_lnz.append(kur(lnz))
    kurtosis_wrx.append(kur(wrx)); kurtosis_wry.append(kur(wry)); kurtosis_wrz.append(kur(wrz))
    kurtosis_mx.append(kur(mx)); kurtosis_my.append(kur(my)); kurtosis_mz.append(kur(mz))
    kurtosis_gx.append(kur(gx)); kurtosis_gy.append(kur(gy)); kurtosis_gz.append(kur(gz))
    
    DFA_lnx.append(dfa(lnx)); DFA_lny.append(dfa(lny)); DFA_lnz.append(dfa(lnz))
    DFA_wrx.append(dfa(wrx)); DFA_wry.append(dfa(wry)); DFA_wrz.append(dfa(wrz))
    DFA_mx.append(dfa(mx)); DFA_my.append(dfa(my)); DFA_mz.append(dfa(mz))
    DFA_gx.append(dfa(gx)); DFA_gy.append(dfa(gy)); DFA_gz.append(dfa(gz))
    
    Variance_lnx.append(var(lnx)); Variance_lny.append(var(lny)); Variance_lnz.append(var(lnz))
    Variance_wrx.append(var(wrx)); Variance_wry.append(var(wry)); Variance_wrz.append(var(wrz))
    Variance_mx.append(var(mx)); Variance_my.append(var(my)); Variance_mz.append(var(mz))
    Variance_gx.append(var(gx)); Variance_gy.append(var(gy)); Variance_gz.append(var(gz)) 
    
    PSD_lnx.append(psd(lnx)); PSD_lny.append(psd(lny)); PSD_lnz.append(psd(lnz))
    PSD_wrx.append(psd(wrx)); PSD_wry.append(psd(wry)); PSD_wrz.append(psd(wrz))
    PSD_mx.append(psd(mx)); PSD_my.append(psd(my)); PSD_mz.append(psd(mz))
    PSD_gx.append(psd(gx)); PSD_gy.append(psd(gy)); PSD_gz.append(psd(gz))

    Entropy_lnx.append(ent(lnx)); Entropy_lny.append(ent(lny)); Entropy_lnz.append(ent(lnz))
    Entropy_wrx.append(ent(wrx)); Entropy_wry.append(ent(wry)); Entropy_wrz.append(ent(wrz))
    Entropy_mx.append(ent(mx)); Entropy_my.append(ent(my)); Entropy_mz.append(ent(mz))
    Entropy_gx.append(ent(gx)); Entropy_gy.append(ent(gy)); Entropy_gz.append(ent(gz))
    
    Lyapunov_lnx.append(lya(lnx)); Lyapunov_lny.append(lya(lny)); Lyapunov_lnz.append(lya(lnz))
    Lyapunov_wrx.append(lya(wrx)); Lyapunov_wry.append(lya(wry)); Lyapunov_wrz.append(lya(wrz))
    Lyapunov_mx.append(lya(mx)); Lyapunov_my.append(lya(my)); Lyapunov_mz.append(lya(mz))
    Lyapunov_gx.append(lya(gx)); Lyapunov_gy.append(lya(gy)); Lyapunov_gz.append(lya(gz))

    mad_lnx.append(meanabsdev(lnx)); mad_lny.append(meanabsdev(lny)); mad_lnz.append(meanabsdev(lnz))
    mad_wrx.append(meanabsdev(wrx)); mad_wry.append(meanabsdev(wry)); mad_wrz.append(meanabsdev(wrz))
    mad_mx.append(meanabsdev(mx)); mad_my.append(meanabsdev(my)); mad_mz.append(meanabsdev(mz))
    mad_gx.append(meanabsdev(gx)); mad_gy.append(meanabsdev(gy)); mad_gz.append(meanabsdev(gz))
    
features = pd.DataFrame()

features['Mean_frequency_lnx'] = Meanfrequency_lnx; features['Mean_frequency_lny'] = Meanfrequency_lny; features['Mean_frequency_lnz'] = Meanfrequency_lnz
features['Mean_frequency_wrx'] = Meanfrequency_wrx; features['Mean_frequency_wry'] = Meanfrequency_wry; features['Mean_frequency_wrz'] = Meanfrequency_wrz
features['Mean_frequency_mx'] = Meanfrequency_mx; features['Mean_frequency_my'] = Meanfrequency_my; features['Mean_frequency_mz'] = Meanfrequency_mz
features['Mean_frequency_gx'] = Meanfrequency_gx; features['Mean_frequency_gy'] = Meanfrequency_gy; features['Mean_frequency_gz'] = Meanfrequency_gz

features['RMS_lnx'] = RMS_lnx; features['RMS_lny'] = RMS_lny; features['RMS_lnz'] = RMS_lnz
features['RMS_wrx'] = RMS_wrx; features['RMS_wry'] = RMS_wry; features['RMS_wrz'] = RMS_wrz
features['RMS_mx'] = RMS_mx; features['RMS_my'] = RMS_my; features['RMS_mz'] = RMS_mz
features['RMS_gx'] = RMS_gx; features['RMS_gy'] = RMS_gy; features['RMS_gz'] = RMS_gz

features['Kurtosis_lnx'] = kurtosis_lnx; features['Kurtosis_lny'] = kurtosis_lny; features['Kurtosis_lnz'] = kurtosis_lnz
features['Kurtosis_wrx'] = kurtosis_wrx; features['Kurtosis_wry'] = kurtosis_wry; features['Kurtosis_wrz'] = kurtosis_wrz
features['Kurtosis_mx'] = kurtosis_mx; features['Kurtosis_my'] = kurtosis_my; features['Kurtosis_mz'] = kurtosis_mz
features['Kurtosis_gx'] = kurtosis_gx; features['Kurtosis_gy'] = kurtosis_gy; features['Kurtosis_gz'] = kurtosis_gz

features['DFA_lnx'] = DFA_lnx; features['DFA_lny'] = DFA_lny; features['DFA_lnz'] = DFA_lnz
features['DFA_wrx'] = DFA_wrx; features['DFA_wry'] = DFA_wry; features['DFA_wrz'] = DFA_wrz
features['DFA_mx'] = DFA_mx; features['DFA_my'] = DFA_my; features['DFA_mz'] = DFA_mz
features['DFA_gx'] = DFA_gx; features['DFA_gy'] = DFA_gy; features['DFA_gz'] = DFA_gz

features['Variance_lnx'] = Variance_lnx; features['Variance_lny'] = Variance_lny; features['Variance_lnz'] = Variance_lnz
features['Variance_wrx'] = Variance_wrx; features['Variance_wry'] = Variance_wry; features['Variance_wrz'] = Variance_wrz
features['Variance_mx'] = Variance_mx; features['Variance_my'] = Variance_my; features['Variance_mz'] = Variance_mz
features['Variance_gx'] = Variance_gx; features['Variance_gy'] = Variance_gy; features['Variance_gz'] = Variance_gz

features['PSD_lnx'] = PSD_lnx; features['PSD_lny'] = PSD_lny; features['PSD_lnz'] = PSD_lnz
features['PSD_wrx'] = PSD_wrx; features['PSD_wry'] = PSD_wry; features['PSD_wrz'] = PSD_wrz
features['PSD_mx'] = PSD_mx; features['PSD_my'] = PSD_my; features['PSD_mz'] = PSD_mz
features['PSD_gx'] = PSD_gx; features['PSD_gy'] = PSD_gy; features['PSD_gz'] = PSD_gz

features['Entropy_lnx'] = Entropy_lnx; features['Entropy_lny'] = Entropy_lny; features['Entropy_lnz'] = Entropy_lnz
features['Entropy_wrx'] = Entropy_wrx; features['Entropy_wry'] = Entropy_wry; features['Entropy_wrz'] = Entropy_wrz
features['Entropy_mx'] = Entropy_mx; features['Entropy_my'] = Entropy_my; features['Entropy_mz'] = Entropy_mz
features['Entropy_gx'] = Entropy_gx; features['Entropy_gy'] = Entropy_gy; features['Entropy_gz'] = Entropy_gz

features['Lyapunov_exponent_lnx'] = Lyapunov_lnx; features['Lyapunov_exponent_lny'] = Lyapunov_lny; features['Lyapunov_exponent_lnz'] = Lyapunov_lnz
features['Lyapunov_exponent_wrx'] = Lyapunov_wrx; features['Lyapunov_exponent_wry'] = Lyapunov_wry; features['Lyapunov_exponent_wrz'] = Lyapunov_wrz
features['Lyapunov_exponent_mx'] = Lyapunov_mx; features['Lyapunov_exponent_my'] = Lyapunov_my; features['Lyapunov_exponent_mz'] = Lyapunov_mz
features['Lyapunov_exponent_gx'] = Lyapunov_gx; features['Lyapunov_exponent_gy'] = Lyapunov_gy; features['Lyapunov_exponent_gz'] = Lyapunov_gz

features['MAD_lnx'] = mad_lnx; features['MAD_lny'] = mad_lny; features['MAD_lnz'] = mad_lnz
features['MAD_wrx'] = mad_wrx; features['MAD_wry'] = mad_wry; features['MAD_wrz'] = mad_wrz
features['MAD_mx'] = mad_mx; features['MAD_my'] = mad_my; features['MAD_mz'] = mad_mz
features['MAD_gx'] = mad_gx; features['MAD_gy'] = mad_gy; features['MAD_gz'] = mad_gz

features.to_csv('mmmm.csv',mode='a',index = False)