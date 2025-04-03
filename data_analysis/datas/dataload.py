import random
from unittest.mock import inplace

import keras.src.saving
import pandas as pd
import numpy as np
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from separate_training import *
from userUtils.query import save_to_sql4


def create_sequences(X, y, t_step=30, lag=0):
    Xs, ys = [], []
    for i in range(len(X) - t_step - lag):
        Xs.append(X[i:i + t_step])
        ys.append(y[i + t_step])
    return np.array(Xs), np.array(ys)

max_model=keras.src.saving.load_model('guangzhoumaxtemp_lstm.h5')
min_model=keras.src.saving.load_model('guangzhoumintemp_lstm.h5')

for i in range(5):
    try:
        a=remove('广州')
        t_step=30
        X1=a[0]
        X11=X1[:,:28]
        X12=np.hstack((X1,np.zeros((len(X1),1))))
        print(X1)
        X2=b[1]
        y1=a[2]
        y2=b[3]
        X_lstm_all_min,y_lstm_all_min=create_sequences(X1,y1,t_step)
        X_lstm_all_max,y_lstm_all_max=create_sequences(-X1,-y1,t_step)
        temp_max=y1[:-32].max(axis=0)
        temp_min2=y2[:-32].min(axis=0)
        scaler1=1/(y1[:-32].max(axis=0)-y1[:-32].min(axis=0))
        scaler2=1/(y2[:-32].max(axis=0)-y2[:-32].min(axis=0))
        # print(X1,X1.shape)
        # print(X_lstm_all_min[-30])
        error=0

        try:
            X_lstm_all_min, y_lstm_all_min = create_sequences(X1, y1, t_step)
            X_lstm_all_min,X_lstm_all_max,result=pre(X_lstm_all_min,X_lstm_all_max,min_model,max_model,temp_max,temp_min2,scaler2,scaler1)
        except:
            try:
                X_lstm_all_min, y_lstm_all_min = create_sequences(X11, y1, t_step)
                X_lstm_all_min, X_lstm_all_max, result = pre(X_lstm_all_min, X_lstm_all_max, min_model, max_model, temp_max,temp_min2, scaler2, scaler1)
            except:
                X_lstm_all_min, y_lstm_all_min = create_sequences(X12, y1, t_step)
                X_lstm_all_min, X_lstm_all_max, result = pre(X_lstm_all_min, X_lstm_all_max, min_model, max_model, temp_max,temp_min2, scaler2, scaler1)
    except:
        error=1
    if error==0:
        break

# print(result)


maximum=[]
minimum=[]
for i in range(30):
    minimum.append(result[i][0][0][0]+50)
    maximum.append(result[i][1][0][0]+6)
    # save_to_sql4(i+1,result[i][1][0][0]+6,result[i][0][0][0]+50)
plt.plot(maximum,label='max')
plt.plot(minimum,label='min')
plt.legend()
plt.show()


