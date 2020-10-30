from google.colab import drive
drive.mount("/content/drive")

!unzip ./drive/My\ Drive/yelp/input.zip

import pandas as pd
import io

review = pd.read_csv("input.csv")
rows = review.shape[0]
cols = ['star','text']
review.columns = cols
review.head()

!pip install vaderSentiment

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

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


data = pd.DataFrame(review[['text','star']])
data.columns = ['summary','rating']

data['summary'] = data['summary'].apply(lambda x: x.lower())
data['summary'] = data['summary'].apply((lambda x: re.sub('[^a-zA-z0-9\s]',' ',x)))

#stopWords = set(['the','and','a','i','to','was','it','in','is','that','my','we','this','they','you','have','had','were','are','me','there','be','our','their','an','been','your','she','he','will','would','us','can','them','im','could','went','her','was','am','those'])
from copy import deepcopy
rows = len(data['rating'])
print(rows)
words = 0
sntncs = 0
brk = ''

inputs = []

sid_obj = SentimentIntensityAnalyzer() 
cnst = (sid_obj.polarity_scores(" "))['compound']

for i in range(rows):
  wds = data['summary'][i].split(' ')
  txt = ""
  for wd in wds:
    snt = wd.split('\\n')
    for s in snt:
      if len(s)>0:
        txt += (" "+s)
  data['summary'][i] = deepcopy(txt)
  
  num_words = 2500
tokenizer = Tokenizer(num_words = num_words, split=' ')
tokenizer.fit_on_texts(data['summary'].values)

X = tokenizer.texts_to_sequences(data['summary'].values)
X = pad_sequences(X)

h = 0
vader_dc = {}
vader_dc[0] = " "
for d in tokenizer.word_index:
  if h>2500:
    break
  vader_dc[h+1] = str(deepcopy(d))
  h += 1
  
  Xv = []
for i in range(rows):
  lst = []
  ln = len(X[i])
  for j in range(4,ln):
    txt = ""
    for k in range(5):
      txt = vader_dc[X[i][j-k]] + " "+ txt
    lst.append((sid_obj.polarity_scores(txt))['compound'])
  Xv.append(np.asarray(lst))
Xv = np.asarray(Xv)
print(Xv.shape)

#building the model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Embedding, LSTM, Dropout, Input, Reshape, concatenate
from tensorflow.keras import Model

embed_dim = 128
num_filter = 1
filter_sz = (5,1)
lstm_out = 500
input_shape = (X.shape[1])
vader_dim = (Xv.shape[1])

def custom_loss_function(actual,prediction):
  loss = (abs(actual-prediction)*2.0)**2.0
  return loss

input_layer = Input(shape=input_shape)
vader_layer = Input(shape=vader_dim)

embd_layer = Embedding(num_words, embed_dim)(input_layer)

lstm_side = LSTM(lstm_out)(embd_layer)
lstm_side = Dropout(0.1)(lstm_side)

concat_layer = concatenate([lstm_side , vader_layer] , axis=-1)

mlp_predict = Dense(100,activation='tanh')(concat_layer)
mlp_predict = Dense(100,activation='tanh')(mlp_predict)

mlp_predict = Dense(1,activation='linear')(mlp_predict)

model = Model(inputs=[input_layer,vader_layer],outputs=mlp_predict)

model.compile(loss = custom_loss_function, optimizer='adam',metrics = ['accuracy'])
print(model.summary())

#training the model
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
Y = np.asarray([float(x-1) for x in data['rating']])


X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size=0.20, random_state=20160121,stratify=Y)
Xv_train, Xv_valid, Yv_train, Yv_valid = train_test_split(Xv,Y, test_size=0.20, random_state=20160121,stratify=Y)

batch_size = 128

model.fit([X_train,Xv_train], Y_train, batch_size =batch_size, epochs = 15,  verbose = 1)

#testing the model
y_predict = model.predict([X_valid,Xv_valid])
n = len(y_predict)
for i in range(n):
  if y_predict[i]<=0:
    y_predict[i]=int(0)
  elif y_predict[i]>4:
    y_predict[i]=int(4)
  else:
    y_predict[i] = int(y_predict[i])
  Y_valid[i]= int(Y_valid[i])

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
    plt.xticks(tick_marks, ['0','1','2','3','4'], rotation=45)
    plt.yticks(tick_marks, ['0','1','2','3','4'])
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
