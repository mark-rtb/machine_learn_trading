# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:01:53 2019

@author: марк
"""

import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, SpatialDropout1D, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.layers.wrappers import Bidirectional


dir_name = 'D:\\traiding_nnet\\data\\'
file_name = 'BTCUSDT_2017-08-01_2019-01-23_1m.dat'

def load_json(dir_name, file_name):
    """ Function to read data from a json file.
The input accepts a file name with extension (example: example.json) output read data. """
    file_name = os.path.join(dir_name, file_name)
    try:
        with open(file_name, encoding="utf8") as data_file:
            list_value = json.load(data_file)
    except UnicodeDecodeError:
        with open(file_name) as data_file:
            list_value = json.load(data_file)
    return list_value

data = load_json(dir_name, file_name)


dat_to_analys = pd.DataFrame(data)

#dat_to_analys['time'] = pd.to_datetime(dat_to_analys['time'], errors='ignore', unit='ms')

info = dat_to_analys.head(500)

def preprocessing_dat(load_df, name_open_col, name_close_col):
    
    load_df['trend'] = load_df[name_open_col] - load_df[name_close_col]#[''] * len(info.iloc[:, 1])
    load_df['trend_bin'] = [''] * len(load_df.iloc[:, 1])
    load_df['trend_predict'] = [''] * len(load_df.iloc[:, 1])
    i = 0
    load_df['trend_predict'][i] = 0
    for row in load_df['trend']:
        if row < 0:
            load_df['trend_bin'][i] = 1
            if i+1 < len(load_df['trend_bin']):
                load_df['trend_predict'][i+1] = 1
        elif row > 0:
            load_df['trend_bin'][i]= 2
            if i+1 < len(load_df['trend_bin']):
                load_df['trend_predict'][i+1] =2
        else:
            load_df['trend_bin'][i] = 0
            if i+1 < len(load_df['trend_bin']):
                load_df['trend_predict'][i+1] =0
        i += 1
    
    return load_df

def data_set_create(df, binary_attribute):
    """ A function of training dataset.
    Produces a regularization of the data and the breakdown of the sample 
    into test and training
    the input takes prepared DataFrame and the name of the column with labels.
    The output is ready to feed data into the network."""
    
    df = df.drop('time', 1)
    
    X = df.iloc[:,:6].values 
    y = df[binary_attribute].values

    from sklearn.preprocessing import StandardScaler 
    scale_features_std = StandardScaler() 
    features_train = scale_features_std.fit_transform(X) 

    # Feature scaling with MinMaxScaler 
    from sklearn.preprocessing import MinMaxScaler 
    scale_features_mm = MinMaxScaler() 
    features_train = scale_features_mm.fit_transform(features_train) 

    X_train, X_test, y_train, y_test = train_test_split(features_train, y,
                                                        test_size=0.2,
                                                        random_state=55)

    return X_train, X_test, y_train, y_test


df_info = preprocessing_dat(info, 'open', 'close')

binary_attribute = 'trend_predict'

X_train, X_test, y_train, y_test = data_set_create(df_info, binary_attribute)

maxlen = X_train.shape[1]


callbacks_list = [EarlyStopping(monitor='val_loss', patience=4),
                  ModelCheckpoint(filepath='dense_traiding.h5',
                                  monitor='val_loss', save_best_only=True,)]



# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(Dense(maxlen,input_shape = (maxlen, ), activation = 'relu'))
#model.add(Conv1D(64, 7, activation = 'relu'))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(64, 5, activation = 'relu'))
#model.add(SpatialDropout1D(0.25))
#model.add(GRU(64, return_sequences=True))
#model.add(GRU(32, recurrent_dropout=0.25))
model.add(Dense(21, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Nadam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    batch_size=48, epochs=40,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks_list)


