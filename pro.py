#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scikits.talkbox.features import mfcc
import scipy
from scipy import io
from scipy.io import wavfile
#from sklearn.matrics import confusion_matrix
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from scikits.audiolab import wavread
import glob
import re
import numpy as np
import os
import warnings
import shutil

warnings.filterwarnings('error')
in_files = glob.glob("/home/leo/Documents/GR/soundData/*[!ceps]/yuki*/*.wav")
out_file = glob.glob("/home/leo/Documents/GR/soundData/*[ceps]/")
#print files
deletedir = glob.glob("/home/leo/Documents/GR/soundData/*[ceps]/*")
'''
for name in deletedir:
    shutil.rmtree(name)

for name in in_files:
    #print ("_____________________________________________")
    #print (name)
    filename = name
    audio, fs, enc = wavread(name)
    try:
        ceps, mspec, spec = mfcc(audio, nwin=256, nfft=512, fs=fs, nceps=13)
    except RuntimeWarning as exc:
        #print "AA"
        continue
    #print "BB"
    dataname = name[-25:-4]
    #print dataname
    dirname = name[:-45] + "ceps/" + dataname[:-3] + "/"
    #print name[:-45]
    #print dirname
    try:
        os.mkdir(dirname)
    except:
        print ()
        #print "既にディレクトリが存在するので削除して作り直しました"
        #shutil.rmtree(dirname)
    	#os.mkdir(dirname)

    datapath = dirname + dataname + ".ceps"
    #print datapath
    np.save(datapath,ceps)
'''
X,Y = [],[]
read_path = glob.glob("/home/leo/Documents/GR/soundData/*[ceps]/*/*.ceps.npy")
for name in read_path:
    label = name[33:-50]
    ceps = np.load(name)
    num_ceps = len(ceps)
    X.append(np.mean(ceps[:],axis=0))
    Y.append(label)
x = np.array(X)
y = np.array(Y)
svc = LinearSVC(C=1.0)
x,y = resample(x,y,n_samples=len(y))
svc.fit(x[150:],y[150:])
prediction = svc.predict(x[:150])
print y[:150]
print"#############################"
print prediction
from sklearn.metrics import accuracy_score,classification_report
print accuracy_score(y[:150], prediction)
from sklearn.metrics import classification_report
print classification_report(y[:150], prediction)
