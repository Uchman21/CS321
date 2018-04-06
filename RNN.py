import pandas as pd
from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
import random as rn
rn.seed(0)
import os
os.environ['PYTHONHASHSEED'] = '0'
from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D
from keras.layers import LSTM, Lambda
from keras.layers import TimeDistributed, Bidirectional
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
import re
import keras.callbacks
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from numpy import around
from keras.utils import plot_model






def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71


def striphtml(html):
    p = re.compile(r'<.*?>')
    return p.sub('', html)


def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)


# record history of training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


total = len(sys.argv)
cmdargs = str(sys.argv)


if len(sys.argv) < 2:
  print("Enter Dataset Name!!!")
  exit()

print ("Script name: %s" % str(sys.argv[0]))
print ("Dataset name: %s" % str(sys.argv[1]))

checkpoint = None

if len(sys.argv) == 3:
    if os.path.exists(str(sys.argv[2])):
        print ("Checkpoint : %s" % str(sys.argv[2]))
        checkpoint = str(sys.argv[2])

# data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
data = pd.read_csv("Files/{}.csv".format(str(sys.argv[1])), header=0, delimiter="\t", quoting=3, usecols=[0,3,4])
txt = ''
docs = []
sentences = []
labels = []

for cont, label in zip(data.transformed, data.label):
    # sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean(striphtml(cont)))

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', cont)
    sentences = [sent.lower() for sent in sentences]
    docs.append(sentences)
    labels.append(label)

num_sent = []
for doc in docs:
    num_sent.append(len(doc))
    for s in doc:
        txt += s

chars = set(txt)

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print('Sample doc{}'.format(docs[1200]))

maxlen = 32
max_sentences = 2

X = np.ones((len(docs), max_sentences, maxlen), dtype=np.int64) * -1
y = np.array(labels)

for i, doc in enumerate(docs):
    for j, sentence in enumerate(doc):
        if j < max_sentences:
            for t, char in enumerate(sentence[-maxlen:]):
                X[i, j, (maxlen - 1 - t)] = char_indices[char]

print('Sample chars in X:{}'.format(X[1200, 1]))
print('y:{}'.format(y[1200]))

ids = np.arange(len(X))
np.random.shuffle(ids)

# shuffle
X = X[ids]
y = y[ids]

# X_train = X[:13500]
# X_test = X[13500:]

# y_train = y[:13500]
# y_test = y[13500:]

n_split = 5

sss = StratifiedKFold(n_splits=n_split, random_state=0)

iter=0

for train_index, test_index in sss.split(X, y):
# ...    print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  print(y_test.sum())
  print(y_test.shape)
 # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

  filter_length = [3, 2, 2]
  nb_filter = [64, 64, 64]
  pool_length = 2
  # document input
  document = Input(shape=(max_sentences, maxlen), dtype='int64')
  # sentence input
  in_sentence = Input(shape=(maxlen,), dtype='int64')
  # char indices to one hot matrix, 1D sequence to 2D 
  embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
  # embedded: encodes sentence
  for i in range(len(nb_filter)):
      embedded = Conv1D(filters=nb_filter[i],
                        kernel_size=filter_length[i],
                        padding='valid',
                        activation='relu',
                        kernel_initializer='glorot_normal',
                        strides=1)(embedded)

      embedded = Dropout(0.1)(embedded)
      embedded = MaxPooling1D(pool_size=pool_length)(embedded)

  bi_lstm_sent = \
      Bidirectional(LSTM(64, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(embedded)

  # sent_encode = merge([forward_sent, backward_sent], mode='concat', concat_axis=-1)
  sent_encode = Dropout(0.3)(bi_lstm_sent)
  # sentence encoder
  encoder = Model(inputs=in_sentence, outputs=sent_encode)
  encoder.summary()
  # plot_model(encoder, to_file='Images/enoder2.png')
  # exit()

  encoded = TimeDistributed(encoder)(document)
  # encoded: sentences to bi-lstm for document encoding 
  b_lstm_doc = \
      Bidirectional(LSTM(64, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(encoded)

  output = Dropout(0.3)(b_lstm_doc)
  output = Dense(64, activation='relu')(output)
  output = Dropout(0.3)(output)
  output = Dense(1, activation='sigmoid')(output)

  model = Model(inputs=document, outputs=output)

  model.summary()
  # plot_model(model, to_file='Images/model.png')
  # exit()

  if checkpoint:
      model.load_weights(checkpoint)

  file_name = os.path.basename(sys.argv[0]).split('.')[0]
  check_cb = keras.callbacks.ModelCheckpoint('checkpoints/' + file_name + '.hdf5',
                                             monitor='val_loss',
                                             verbose=0, save_best_only=True, mode='min')
  earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
  history = LossHistory()
  optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
  model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64,
            epochs=30, shuffle=False, callbacks=[earlystop_cb, check_cb])
  # model.fit(X_train, y_train, validation_split=0.1, batch_size=100,
  #           epochs=5, shuffle=True, callbacks=[earlystop_cb, check_cb])

  model.load_weights('checkpoints/' + file_name + '.hdf5')

  if iter == 0:
    # tr_acc = np.array( model.history.history['binary_accuracy'])
    # val_acc = np.array(model.history.history['val_binary_accuracy'])
    # tr_loss = np.array(model.history.history['loss'])
    # val_loss = np.array(model.history.history['val_loss'])
    predY = around(model.predict(X_test))
    acc, precision , recall, f1 = [accuracy_score(y_test, predY), precision_score(y_test, predY, average='binary'), recall_score(y_test, predY, average='binary'), f1_score(y_test, predY, average='binary')]

  else:
    # tr_acc += np.array(model.history.history['binary_accuracy'])
    # val_acc += np.array(model.history.history['val_binary_accuracy'])
    # tr_loss += np.array(model.history.history['loss'])
    # val_loss += np.array(model.history.history['val_loss'])
    predY = around(model.predict(X_test))
    acc += accuracy_score(y_test, predY) 
    precision += precision_score(y_test, predY, average='binary')
    recall += recall_score(y_test, predY, average='binary')
    f1 += f1_score(y_test, predY, average='binary')
  iter += 1


  
print (acc/n_split)
print (precision/n_split)
print (recall/n_split)
print (f1/n_split) 

# print(model.history.keys())
# summarize history for accuracy
# plt.figure(1)
# plt.plot(tr_acc/n_split)
# plt.plot(val_acc/n_split)
# plt.title('{} model accuracy'.format(str(sys.argv[1]).split("_all_")[-1]))
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# # plt.show()
# # summarize history for loss
# plt.figure(2)
# plt.plot(tr_loss/n_split)
# plt.plot(val_loss/n_split)
# plt.title('{} model loss'.format(str(sys.argv[1]).split("_all_")[-1]))
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

print (str(sys.argv[1]))


''' my_df = pd.DataFrame(history.accuracies)
my_df.to_csv('accuracies.csv', index=False, header=False)

my_df = pd.DataFrame(history.losses)
my_df.to_csv('losses.csv', index=False, header=False)'''

'''plt.plot(history.accuracies)
plt.ylabel('Batch_per_epoch')
plt.ylabel('Accuraccy')
plt.show()
plt.plot(history.losses)
plt.ylabel('Batch_per_epoch')
plt.ylabel('Loss')
plt.show()'''


# just showing access to the history object
# print metrics.f1s
# print history.losses
# print history.accuracies
