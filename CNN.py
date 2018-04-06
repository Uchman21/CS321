import pandas as pd
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)
import random as rn
rn.seed(2018)

from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import keras.backend as K
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


import os


def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71


def striphtml(s):
    p = re.compile(r'<.*?>')
    return p.sub('', s)


def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)


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

print('Sample X:{}'.format(X[1200, 0]))
print('y:{}'.format(y[1200]))
# exit()

ids = np.arange(len(X))
np.random.shuffle(ids)

# shuffle
X = X[ids]
y = y[ids]

# X_train = X[:20000]
# X_test = X[22500:]

# y_train = y[:20000]
# y_test = y[22500:]


def char_block(in_layer, nb_filter=(64, 100), filter_length=(5, 5), subsample=(3, 1), pool_length=(2, 2)):
    block = in_layer
    for i in range(len(nb_filter)):

        block = Conv1D(filters=nb_filter[i],
                       kernel_size=filter_length[i],
                       padding='valid',
                       activation='tanh',
                       strides=subsample[i])(block)

        # block = BatchNormalization()(block)
        # block = Dropout(0.1)(block)
        if pool_length[i]:
            block = MaxPooling1D(pool_size=pool_length[i])(block)

    # block = Lambda(max_1d, output_shape=(nb_filter[-1],))(block)
    block = GlobalMaxPool1D()(block)
    block = Dense(128, activation='relu')(block)
    return block


max_features = len(chars) + 1
char_embedding = 40

n_split = 5

sss = StratifiedKFold(n_splits=n_split, random_state=0)

iter=0

for train_index, test_index in sss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(y_test.sum())
    print(y_test.shape)

    document = Input(shape=(max_sentences, maxlen), dtype='int64')
    in_sentence = Input(shape=(maxlen,), dtype='int64')

    embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)

    block2 = char_block(embedded, (128, 256), filter_length=(5, 5), subsample=(1, 1), pool_length=(2, 2))
    block3 = char_block(embedded, (192, 320), filter_length=(7, 5), subsample=(1, 1), pool_length=(2, 2))
    block4 = char_block(embedded, (64, 64), filter_length=(3, 3), subsample=(1, 1), pool_length=(2, 2))

    sent_encode = concatenate([block2, block3, block4], axis=-1)
    # sent_encode = Dropout(0.2)(sent_encode)

    encoder = Model(inputs=in_sentence, outputs=sent_encode)
    encoder.summary()
    # plot_model(encoder, to_file='Images/enoder.png')
    # exit()

    encoded = TimeDistributed(encoder)(document)

    lstm_h = 64

    lstm_layer = LSTM(lstm_h, return_sequences=True, dropout=0.3, recurrent_dropout=0.1, implementation=0)(encoded)
    lstm_layer2 = LSTM(lstm_h, return_sequences=False, dropout=0.3, recurrent_dropout=0.1, implementation=0)(lstm_layer)

    # output = Dropout(0.2)(bi_lstm)
    output = Dense(1, activation='sigmoid')(lstm_layer2)

    model = Model(outputs=output, inputs=document)

    model.summary()
    # plot_model(model, to_file='Images/model2.png')

    if checkpoint:
        model.load_weights(checkpoint)

    file_name = os.path.basename(sys.argv[0]).split('.')[0]

    check_cb = keras.callbacks.ModelCheckpoint('checkpoints/' + file_name + '.hdf5',
                                               monitor='val_loss',
                                               verbose=0, save_best_only=True, mode='min')

    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=500, epochs=10, shuffle=True, callbacks=[check_cb, earlystop_cb])

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

