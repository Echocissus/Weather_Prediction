from unittest.mock import inplace
import pandas as pd
import numpy as np
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras import optimizers
from pandas import isnull
from pyexpat import features
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics
from tensorflow.python.ops.numpy_ops.np_math_ops import average
from tensorflow.python.training import adam
from tensorflow.python.training.adam import AdamOptimizer
from torch.ao.quantization.utils import activation_dtype
from torch.nn import Sequential
from keras import Sequential
from matplotlib import pyplot as plt

from userUtils import query
import sqlalchemy
from sqlalchemy import create_engine
from pymysql import *
import json
import pandas as pd
from sklearn import preprocessing

from tensorflow.python.keras.losses import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel



# print(max_features[0])
def removal(city):
    date_list1 = pd.DataFrame(list(pd.date_range('2022-01-01', '2025-03-29', freq='D')),
                             index=list(pd.date_range('2022-01-01', '2025-03-29', freq='D')))
    date_list2 = pd.DataFrame(list(pd.date_range('2015-01-01', '2025-03-25', freq='D')),
                              index=list(pd.date_range('2015-01-01', '2025-03-25', freq='D')))
    a = pd.read_csv('{}data_for_prediction.csv'.format(city))
    a = a.set_index('Unnamed: 0', drop=True)
    # print(a.info(),a.columns)
    a.head()
    a = pd.get_dummies(a)
    # print(a)
    l1 = np.array(a['max_temp'].to_list())
    l2 = np.array(a['min_temp'].to_list())
    delta=l1-l2
    # a1_max = a.drop(columns=['min_temp'], axis=1)
    # a1_min = a.drop(columns=['max_temp'], axis=1)
    c = []
    b = []
    # print(a['max_temp'])

    temp1=a['max_temp'].values
    temp2=a['min_temp'].values
    avg=(temp1+temp2)/2
    a =  a.drop(columns=['year'],axis=1)
    a = a.drop(columns=['month'], axis=1)
    a = a.drop(columns=['day'], axis=1)
    a = a.drop(columns=['min_temp'], axis=1)
    a = a.drop(columns=['max_temp'], axis=1)
    a=a.values
    for i in range(len(a[0])):
        try:
            c.append(float(np.var(a,axis=0)[i]))
            b.append(float(np.var(a,axis=0)[i]))
            if c[i]<=(0.9*(1-0.9)): #0.8是出现同一个值的概率
                a=np.delete(a,i,axis=1)
            if b[i]<=(0.9*(1-0.9)):
                a=np.delete(a,i,axis=1)
        except:
            pass
            # print(a)
            # print(ad)
            # print(b)
            # print(bd)
            # print(max_features[0])
    etc=ExtraTreesClassifier()
    etc1=etc.fit(a,l1)
    etc2=etc.fit(a,l2)
    # print(etc1.feature_importances_)
    model=SelectFromModel(etc1,prefit=True)
    a1=model.transform(a)
    model2 = SelectFromModel(etc2, prefit=True)
    a2 = model2.transform(a)
    a1=np.hstack((temp1.reshape(len(temp1),1),a1))
    a1 = np.hstack((temp2.reshape(len(temp2), 1), a1))
    a1=np.hstack((delta.reshape(len(temp2), 1), a1))
    # a1 = np.hstack((avg.reshape(len(temp1), 1), a1))
    a2 = np.hstack((temp1.reshape(len(temp1), 1), a2))
    a2=np.hstack((temp2.reshape(len(temp2),1),a2))
    return [a1,a2,l1,l2]





b=removal('广州')
X1=-b[0]
X2=b[1]
y1=-b[2]
y2=b[3]




#开始预测
def predict2(X1,X2,y1,y2):


    def model_construction_min(i=1, j=50, k=1):
        model = Sequential()
        model.add(LSTM(j, return_sequences=(i > 1), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.3))
        for layers in range(1, i):
            model.add(LSTM(j, return_sequences=(layers < i - 1)))
            model.add(Dropout(0.3))
        for layers in range(k):
            model.add(Dense(units=j, activation='relu'))
        model.add(Dense(1))
        return model



    # scaler=MinMaxScaler()
    # X1=scaler.fit_transform(X1)
    # X2=scaler.fit_transform(X2)
    # print(X1,X1.shape)
    temp_min1=y1[:-32].min(axis=0)
    temp_min2=y2[:-32].min(axis=0)
    temp_max=y1[:-32].max(axis=0)
    scaler1=1/(y1[:-32].max(axis=0)-y1[:-32].min(axis=0))
    scaler2=1/(y2[:-32].max(axis=0)-y2[:-32].min(axis=0))
    y1=(y1-temp_min1)*scaler1
    y2=(y2-temp_min2)*scaler2


    # print(type(a[2]))
    #创建最终预测窗口:

    #创建滑动窗口
    def create_sequences(X,y,t_step=30,lag=0):
        Xs,ys=[],[]
        for i in range(len(X)-t_step-lag):
            Xs.append(X[i:i+t_step])
            ys.append(y[i+t_step])
        return np.array(Xs),np.array(ys)



    t_step=30


    #预测未来30天

    X_lstm,y_lstm=create_sequences(X1,y1,t_step,30)
    X_lstm1,y_lstm1=create_sequences(X2,y2,t_step,30)
    X_lstm_all_max,y_lstm_all_max=create_sequences(X1,y1,t_step)
    # print(X_lstm_all_max[-30,0],X_lstm_all_max[-30],X_lstm_all_max[-30].shape)

    print('sort..')#划分数据集

    X_train,X_temp,y_train,y_temp=train_test_split(X_lstm,y_lstm,test_size=0.2,shuffle=False)
    X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,shuffle=False)
    X_train1,X_temp1,y_train1,y_temp1=train_test_split(X_lstm1,y_lstm1,test_size=0.2,shuffle=False)
    X_val1,X_test1,y_val1,y_test1=train_test_split(X_temp1,y_temp1,test_size=0.5,shuffle=False)

    RMSE=[]
    # print(X_lstm1,X_lstm1.shape)

    #构建模型
    print('modeling...')
    model=model_construction_min(3,200,2)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) #编译
    #早停
    earlystopping=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
    #train
    # history=model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val,y_val),callbacks=[earlystopping])
    history=model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val,y_val),callbacks=[earlystopping])

    # predictions = model.predict(X_test)
    # tem_predicted = (predictions / scaler1) + y1.min(axis=0)
    # tem_predicted = -tem_predicted
    # real= (y_test.reshape(-1,1)/scaler1)+y1.min(axis=0)
    # real1=-real[-1]
    # real2=pd.read_csv('{}data.csv'.format(city))['max_temp'].values[0]
    # print(real2)
    # add=real2-real1
    # tem_predicted=tem_predicted+add
    # real=-real+add
    # print(tem_predicted)
    # print(real)
    # mse = mean_squared_error(real, tem_predicted)
    # rmse = np.sqrt(mse)
    # print(rmse,max(rmse),average(rmse))
    # RMSE.append(average(rmse))

    # for i in range(30):
    #     predictions=model.predict(X_end[i].reshape(1,60,27))
    #     tem_predicted = (predictions / scaler1) + temp_min1
    #     predict.append(tem_predicted[0,0])

    # real= (y_test1.reshape(-1,1)/scaler2)+y2.min(axis=0)
    # print(predict)
    # pre=pd.DataFrame(predict,index=list(range(30)))
    # pre.to_csv('广州预测.csv')
    # print('*'*100)


    # 求RMSE:
    # mse=mean_squared_error(real,predict)
    # rmse=np.sqrt(mse)
    # print(rmse)

    # 画损失function
    # plt.figure(figsize=(20,8),dpi=100)
    # plt.plot(history.history['loss'],label='Training LOSS')
    # plt.plot(history.history['val_loss'],label='VAL_LOSS')
    # plt.legend()
    # plt.show()


    #写文件

    # model.save('guangzhoumaxtemp_lstm.h5')
    return model,X_lstm,scaler1,temp_max,X_lstm_all_max

# print(df)
#画预测/真实值对比function:
# plt.figure(figsize=(20,8),dpi=100)
# plt.plot(real,label='real')
# plt.plot(tem_predicted,label='prediction')
# plt.legend()
# plt.show()

# md=predict2('广州')