# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import numpy as np
from keras import layers
from sklearn.metrics import r2_score
import numpy as np
import xgboost
from lssvr import LSSVR
import random
import tensorflow as tf
import os
def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


setup_seed(1)

def gru(deliver_data,length,num):
    #print("选用的模型是GRU")
    train_data = deliver_data[0]
    val_data = deliver_data[1]

    seq_length = length
    train_x = train_data[:, :seq_length, :]
    train_y = train_data[:, -1, -1]

    test_x = val_data[:, :seq_length, :]
    test_y = val_data[:, -1, -1]

    mean = train_x.mean(axis=0)  # 均值
    std = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    batch_size = 64

    model = keras.Sequential()
    model.add(layers.GRU(num, input_shape=(seq_length, 1)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])
    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=50,  # 迭代次数30

                       verbose=0)
    outcome_y = model.predict(test_x)
    train_yp=model.predict(train_x)
    print("GRU模型的R2:" + str(r2_score(test_y, outcome_y)))

    return outcome_y, test_y,train_yp,train_y

def lstm(deliver_data,length,num):
    setup_seed(1)
    #print("选用的模型是LSTM")
    train_data=deliver_data[0]
    val_data=deliver_data[1]

    seq_length=length
    train_x = train_data[:, :seq_length, :]
    train_y = train_data[:, -1, -1]

    test_x = val_data[:, :seq_length, :]
    test_y = val_data[:, -1, -1]




    mean = train_x.mean(axis=0)  # 均值
    std = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    batch_size = 64

    model = keras.Sequential()
    model.add(layers.LSTM(num, input_shape=(seq_length, 1)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])
    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=50,  # 迭代次数30

                       verbose=0)
    outcome_y = model.predict(test_x)
    train_yp = model.predict(train_x)
    print("LSTM模型的R2:" + str(r2_score(test_y, outcome_y)))

    return outcome_y, test_y,train_yp,train_y

def cnn(deliver_data,length,num):
    #print("选用的模型是CNN")
    train_data = deliver_data[0]
    val_data = deliver_data[1]

    seq_length = length
    train_x = train_data[:, :seq_length, :]
    train_y = train_data[:, -1, -1]

    test_x = val_data[:, :seq_length, :]
    test_y = val_data[:, -1, -1]

    mean = train_x.mean(axis=0)  # 均值
    std = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    batch_size = 64
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(train_x.shape[1], train_x.shape[2])))

    model.add(layers.Conv1D(num,2))
    model.add(layers.Flatten()),
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])
    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=50,  # 迭代次数30

                       verbose=0)
    outcome_y = model.predict(test_x)
    train_yp = model.predict(train_x)
    print("CNN模型的R2:" + str(r2_score(test_y, outcome_y)))

    return outcome_y, test_y,train_yp,train_y

def bp(deliver_data,length,num):
    #print("选用的模型是BP")
    train_data = deliver_data[0]
    val_data = deliver_data[1]

    seq_length = length
    train_x = train_data[:, :seq_length, :]
    train_y = train_data[:, -1, -1]

    test_x = val_data[:, :seq_length, :]
    test_y = val_data[:, -1, -1]

    mean = train_x.mean(axis=0)  # 均值
    std = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    batch_size = 64

    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(layers.Flatten()),
    model.add(layers.Dense(num))

    model.add(layers.Dense(1))
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])
    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=50,  # 迭代次数30

                       verbose=0)
    outcome_y = model.predict(test_x)
    train_yp = model.predict(train_x)
    print("BP模型的R2:" + str(r2_score(test_y, outcome_y)))

    return outcome_y, test_y,train_yp,train_y

def xgb(deliver_data,length,num):
    #print("选用的模型是XGBoost")
    train_data = deliver_data[0]
    val_data = deliver_data[1]

    seq_length = length
    train_x = train_data[:, :seq_length, -1]
    train_y = train_data[:, -1, -1]

    test_x = val_data[:, :seq_length, -1]
    test_y = val_data[:, -1, -1]

    mean = train_x.mean(axis=0)  # 均值
    std = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    model = xgboost.XGBRegressor().fit(train_x, train_y)

    outcome_y = model.predict(test_x)
    train_yp = model.predict(train_x)

    print("XGB模型的R2:" + str(r2_score(test_y, outcome_y)))
    return outcome_y, test_y,train_yp,train_y

def svr(deliver_data,length,num):
    #print("选用的模型是LSSVR")
    train_data = deliver_data[0]
    val_data = deliver_data[1]

    seq_length = length
    train_x = train_data[:, :seq_length, -1]
    train_y = train_data[:, -1, -1]

    test_x = val_data[:, :seq_length, -1]
    test_y = val_data[:, -1, -1]

    mean = train_x.mean(axis=0)  # 均值
    std = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std


    model = LSSVR(kernel='rbf', gamma=0.01).fit(train_x, train_y)

    outcome_y = model.predict(test_x)
    train_yp = model.predict(train_x)
    print("LSSVR模型的R2:"+str(r2_score(test_y, outcome_y)))

    return outcome_y, test_y,train_yp,train_y
