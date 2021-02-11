# Artificial-neural-network-Car-Sales
Written in python
# loading all required packages
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import re
import time
import warnings
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
from mlxtend.classifier import StackingClassifier

# Loading the dataset
df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO -8859-1') # This encoding is necessary because the dataset has special characters like @
print(df.head(20))
print()
print(df.tail(20))

# Visualizing the dataset
print(sns.pairplot(df))
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

# cleaning the dataset

x = df.drop(['Customer Name','Customer e-mail','Country', 'Car Purchase Amount'], axis=1)
predict = df['Car Purchase Amount']
y = predict
print(x.shape, len(x))
print()
print(y.shape,len(y))

# Normalizing the data:
scaler = MinMaxScaler()
scaled_X= scaler.fit_transform(x)
print(scaler.data_max_)
print(scaler.data_min_)

# reshaping the label and normalizing it
y= y.values.reshape(-1,1)
scaled_y = scaler.fit_transform(y)
print(scaled_y)
print(scaled_y.shape)

# Training the Model
train_x,test_x,train_y,test_y = sklearn.model_selection.train_test_split(scaled_X,scaled_y,test_size=0.1)

print(train_x.shape)
print(train_y.shape)

model = keras.Sequential()
model.add(Dense(40, input_shape =(5,), activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(1,activation='linear'))

# Summary of model
print(model.summary())

model.compile(optimizer= 'adam', loss= 'mean_squared_error', metrics=['accuracy'])

# Fitting the model
model_hist = model.fit(train_x,train_y,epochs=100,verbose=2,validation_split=0.2, batch_size=25)

# Evaluating the model
eval = model.evaluate(test_x,test_y)
print(eval)

print(model_hist.history.keys())

# Visualizing the accuracies and losses of the model
fig = plt.figure()

plt.subplot(2,2,1)
plt.plot(model_hist.history['loss'])
plt.title('Loss graph')
plt.xlabel('Number of epochs')
plt.ylabel('Model loss')

plt.subplot(2,2,2)
plt.plot(model_hist.history['val_loss'])
plt.title('validation loss graph')
plt.xlabel('Number of epochs')
plt.ylabel('Validation loss')


plt.subplot(2,2,3)
plt.plot(model_hist.history['accuracy'])
plt.title('accuracy graph')
plt.xlabel('Number of epochs')
plt.ylabel('Model\'s accuarcy')


plt.subplot(2,2,4)
plt.plot(model_hist.history['val_accuracy'])
plt.title('val_accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Model\'s validation accuracy')


plt.tight_layout(pad=2)

plt.show()

# To make predictions
prediction = model.predict(test_x)
for i in enumerate(prediction):
    print('The expected purchase amount is', prediction, 'the features used is making predictions are:',test_x)


