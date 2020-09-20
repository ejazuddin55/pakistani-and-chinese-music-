import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Keras
import keras
from keras import models
from keras import layers
from  sklearn import svm
import sklearn
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
songs = ' classical country minge pak pop'.split()
for g in songs:
    for filename in os.listdir(f'./songs/{g}'):
        songname = f'./songs/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=10 )
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

# reading dataset from csv

data = pd.read_csv('data.csv', error_bad_lines=False, quoting=csv.QUOTE_NONE, warn_bad_lines=False)
data.head()
print(data)

# Dropping unneccesary columns
data = data.drop(['filename'], axis=1)
data.head()

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
print(y)

# normalizing
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)
X_train, X_test, y_train, y_test =sklearn.model_selection.train_test_split(X, y, test_size=0.2)

#print(x_train, y_train)
label = ['minge' 'pak' 'classical ' 'country ' 'pop']
k_range = range(1, 26)
score = {} 
score_list= []
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import Pipeline
for k in k_range: knn = KNeighborsClassifier(n_neighbors= 9) 
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test) 
score[k] = sklearn.metrics.accuracy_score(y_test, y_pred) 
score_list.append(sklearn.metrics.accuracy_score(y_test, y_pred)) 
plt.plot(k_range,score_list)
plt.xlabel(' value of k for KNN') 
plt.ylabel('testing accuracy')
plt.show() 
print(score[k])
names = ["minge", "pak"]
predicted = knn.predict(X_test) 
print(predicted)

