#a colab file structure......

from google.colab import drive
drive.mount("/content/drive")

!unzip ./drive/My\ Drive/yelp/input.zip

import pandas as pd
import io

review = pd.read_csv("input.csv")
review.head()
rows = review.shape[0]

cols = ['star','text']
review.columns = cols

review.head()

import numpy as np 
import pandas as pd
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM , Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import pandas as pd


data = pd.DataFrame(review[['text','star']])
data.columns = ['text','stars']

data['sentiment']=[x for x in data['stars']]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
print(data)
num_words = 2500
tokenizer = Tokenizer(num_words = num_words, split=' ')
tokenizer.fit_on_texts(data['text'].values)
#print(tokenizer.word_index)  # To see the dicstionary
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Embedding, LSTM, Dropout, Input, Reshape, concatenate
from tensorflow.keras import Model

#bilding model
embed_dim = 128
num_filter = 4
filter_sz = (5,64)
pool_sz = (57,13)
lstm_out = 300
input_shape = (X.shape[1])

def custom_loss_function(actual,prediction):
  loss = (abs(actual-prediction)*2.5)**2.0
  return loss

input_layer = Input(shape=input_shape)
embd_layer = Embedding(num_words, embed_dim)(input_layer)

#embd_layer = Dropout(0.2)(input_layer)

cnn_side = Reshape((X.shape[1],embed_dim,1))(embd_layer)
cnn_side = Conv2D(num_filter, filter_sz)(cnn_side)
cnn_side = MaxPooling2D(pool_size=pool_sz)(cnn_side)
cnn_side = Flatten()(cnn_side)
cnn_side = Dropout(0.1)(cnn_side)

lstm_side = LSTM(lstm_out)(embd_layer)
lstm_side = Dropout(0.1)(lstm_side)

concat_layer = concatenate([cnn_side , lstm_side] , axis=-1)

mlp_predict = Dense(100,activation='softsign')(concat_layer)
mlp_predict = Dense(100,activation='tanh')(mlp_predict)

mlp_predict = Dense(1,activation='linear')(mlp_predict)

model = Model(inputs=input_layer,outputs=mlp_predict)

model.compile(loss = custom_loss_function, optimizer='adam',metrics = ['accuracy'])
print(model.summary())

from sklearn.model_selection import StratifiedKFold
Y = np.asarray([float(x-1) for x in data['sentiment']])
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size=0.20, random_state=20160121,stratify=Y)


#Here we train the Network.
batch_size = 128

model.fit(X_train, Y_train, batch_size =batch_size, epochs = 5,  verbose = 1)

#test the model
y_predict = model.predict(X_valid)
n = len(y_predict)
for i in range(n):
  if y_predict[i]<=0:
    y_predict[i]=int(0)
  elif y_predict[i]>=4:
    y_predict[i]=int(4)
  else:
    y_predict[i]=int(y_predict[i])
  Y_valid[i] = int(Y_valid[i])
  
  from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
print(metrics.classification_report(Y_valid,y_predict))

import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, ['1','2','3','4','5'], rotation=45)
    plt.yticks(tick_marks, ['1','2','3','4','5'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
cm = confusion_matrix(Y_valid, y_predict)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm)    

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()
