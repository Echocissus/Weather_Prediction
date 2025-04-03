import random
from unittest.mock import inplace
import pandas as pd
import numpy as np
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pandas import isnull
from pyexpat import features
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics
from tensorflow.python.ops.numpy_ops.np_math_ops import average
from tensorflow.python.ops.numpy_ops.np_random import randint
from torch.ao.quantization.utils import activation_dtype
from torch.nn import Sequential
from keras import Sequential
from matplotlib import pyplot as plt
from userUtils import query
import sqlalchemy
from sqlalchemy import create_engine
from pymysql import *
import json
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from training import *



# print(max_features[0])
def remove(city):
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
    a = a.drop(columns=['year'], axis=1)
    a = a.drop(columns=['month'], axis=1)
    a = a.drop(columns=['day'], axis=1)
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
    a2 = np.hstack((temp1.reshape(len(temp1), 1), a2))
    a2=np.hstack((temp2.reshape(len(temp2),1),a2))
    return [a1,a2,l1,l2]



a=remove('广州')
X1=a[0]
X2=a[1]
y1=a[2]
y2=a[3]
print(X1.shape)





def predict(X1,X2,y1,y2):
    #开始预测

    # print(X1,X1.shape)
    # scaler = MinMaxScaler()
    # X1 = scaler.fit_transform(X1)
    # X2 = scaler.fit_transform(X2)
    temp_min1=y1[:-32].min(axis=0)
    temp_min2=y2[:-32].min(axis=0)
    scaler1=1/(y1[:-32].max(axis=0)-y1[:-32].min(axis=0))
    scaler2=1/(y2[:-32].max(axis=0)-y2[:-32].min(axis=0))
    y1=(y1-temp_min1)*scaler1
    y2=(y2-temp_min2)*scaler2
    # print(y1)
    # print(y2)
    prediction=[]
    for i in range(30):
        prediction.append(i)


    # print(type(a[2]))

    #创建滑动窗口
    def create_sequences(X,y,t_step=30,lag=0):
        Xs,ys=[],[]
        for i in range(len(X)-t_step-lag):
            Xs.append(X[i:i+t_step])
            ys.append(y[i+t_step])
        return np.array(Xs),np.array(ys)


    t_step=30
    X_lstm_all_min,y_lstm_all_min=create_sequences(X1,y1,t_step)
    X_lstm,y_lstm=create_sequences(X1,y1,t_step,30)
    X_lstm1,y_lstm1=create_sequences(X1,y2,t_step,30)
    # print(X_lstm_all_min[-30,0])

    print('sort..')#划分数据集
    X_train,X_temp,y_train,y_temp=train_test_split(X_lstm,y_lstm,test_size=0.2,shuffle=False)
    X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,shuffle=False)
    X_train1,X_temp1,y_train1,y_temp1=train_test_split(X_lstm1,y_lstm1,test_size=0.2,shuffle=False)
    X_val1,X_test1,y_val1,y_test1=train_test_split(X_temp1,y_temp1,test_size=0.5,shuffle=False)


    #构建模型
    print('modeling...')
    layers=[1,2,3]
    neurons=[50,100,200]
    dense=[1,2,3]

    def model_construction_min(i=1,j=50,k=1):
        model=Sequential()
        model.add(LSTM(j, return_sequences=(i>1), input_shape=(X_train1.shape[1],X_train1.shape[2])))
        model.add(Dropout(0.1))
        for layers in range(1,i):
            model.add(LSTM(j, return_sequences=(layers < i-1)))
            model.add(Dropout(0.1))
        for layers in range(k):
            model.add(Dense(units=j,activation='relu'))
        model.add(Dense(1))
        return model

    def model_construction_max(i=1,j=50,k=1):
        model=Sequential()
        model.add(LSTM(j, return_sequences=(i>1), input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(Dropout(0.1))
        for layers in range(1,i):
            model.add(LSTM(j, return_sequences=(layers < i-1)))
            model.add(Dropout(0.1))
        for layers in range(k):
            model.add(Dense(units=j,activation='relu'))
        model.add(Dense(1))
        return model




    RMSE=[]


    ##遍历找到最佳模型，高温预测最佳LSTM层为2，神经元个数200，dense层一个,低温预测的最佳LSTM层很巧合地与高温预测的相同
    # for i in layers:
    #     for j in neurons:
    #         for k in dense:
    #             model_name='l{}n{}d{}'.format(i,j,k)
    #             print('training {}...'.format(model_name))
    #             model=model_construction_min(i,j,k)
    #             model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) #编译
    #             earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) #早停
    #             history = model.fit(X_train1, y_train1, epochs=100, batch_size=32, validation_data=(X_val1, y_val1),callbacks=[earlystopping])
    #             predictions = model.predict(X_test1)
    #             tem_predicted = (predictions / scaler2) + y2.min(axis=0)
    #             real = (y_test1.reshape(-1, 1) / scaler2) + y2.min(axis=0)
    #             mse = mean_squared_error(real, tem_predicted)
    #             rmse = np.sqrt(mse)
    #             RMSE.append(average(rmse))



    #编译
    # model.summary()


    model=model_construction_min(3,200,2)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) #编译
    #早停
    earlystopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)


    #train
    # history=model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val,y_val),callbacks=[earlystopping])
    history=model.fit(X_train1, y_train1, epochs=100, batch_size=32, validation_data=(X_val1,y_val1),callbacks=[earlystopping])
    # model.save('guangzhoumintemp_lstm.h5')
    return model,X_lstm1,scaler2,temp_min2,X_lstm_all_min



def pre(X_lstm_all_min,X_lstm_all_max,model,max_model,temp_max,temp_min2,scaler2,scaler1):
    result=[]
    min_tem_predicted=[]
    max_tem_predicted=[]
    print('*'*100)
    for i in range(30):
        min_prediction=model.predict(X_lstm_all_min[-(30-i)].reshape(1,30,len(X_lstm_all_min[-30,0])))
        # print( temp_min2, scaler2)
        min_tem_predicted.append((min_prediction / scaler2) + temp_min2)
        max_prediction = max_model.predict(-X_lstm_all_max[-(30-i)].reshape(1, 30, len(X_lstm_all_max[-30, 0])))
        max_tem_predicted.append(abs(max_prediction/scaler1) - temp_max)
        # print(max(max_tem_predicted[i],min_tem_predicted[i])-min(min_tem_predicted[i],min_tem_predicted[i]))
        # print(X_lstm_all_max,X_lstm_all_max[-1],X_lstm_all_max.shape)
        if i<29:
            # print(X_lstm_all_max[-(29 - i - 1),-1, 1])
            # print(X_lstm_all_min[-(29 - i - 1),-1, 1])
            for j in range(i+1):
                if max_tem_predicted[j]>min_tem_predicted[j]:
                    X_lstm_all_min[-(29 - i)][-(i - j+1)][2] = max_tem_predicted[j]
                    X_lstm_all_min[-(29 - i)][-(i - j+1)][1] = min_tem_predicted[j]
                    X_lstm_all_min[-(29 - i)][-(i - j+1)][0] = max_tem_predicted[j] - min_tem_predicted[j]
                    X_lstm_all_max[-(29 - i)][-(i - j+1)][2] = -max_tem_predicted[j]
                    X_lstm_all_max[-(29 - i)][-(i - j+1)][1] = -min_tem_predicted[j]
                    X_lstm_all_max[-(29 - i)][-(i - j+1)][0] = -max_tem_predicted[j] + min_tem_predicted[j]
                else:
                    c=random.random()
                    X_lstm_all_min[-(29 - i)][-(i - j+1)][2] = min_tem_predicted[j]+c
                    X_lstm_all_min[-(29 - i)][-(i - j+1)][1] = min_tem_predicted[j]
                    X_lstm_all_min[-(29 - i)][-(i - j+1)][0] =0
                    X_lstm_all_max[-(29 - i)][-(i - j+1)][2] = -min_tem_predicted[j]-c
                    X_lstm_all_max[-(29 - i)][-(i - j+1)][1] = -min_tem_predicted[j]
                    X_lstm_all_max[-(29 - i)][-(i - j+1)][0] = 0
            print(X_lstm_all_max[-(29-i)])
            # print(X_lstm_all_min[-(30-i)])
            # filedata = pd.read_csv('广州data_for_prediction.csv')
            # filedata['max_temp'][-(30-i-1)]=max_tem_predicted
            # filedata['min_temp'][-(30 - i - 1)] = min_tem_predicted
            # filedata.to_csv('广州data_for_prediction.csv')    #重写文件
        # print(max_prediction,max_tem_predicted[i],min_tem_predicted[i])
        result.append([min(max_tem_predicted[i],min_tem_predicted[i]),max(max_tem_predicted[i],min_tem_predicted[i])])
    return X_lstm_all_min,X_lstm_all_max,result
# model,X_lstm1,scaler2,temp_min2,X_lstm_all_min=predict(a[0],a[1],a[2],a[3])
# max_model,X_lstm,scaler1,temp_max,X_lstm_all_max=predict2(-a[0],a[1],-a[2],a[3])

# X_lstm_all_min,X_lstm_all_max,result=pre(X_lstm_all_min,X_lstm_all_max,model,max_model,temp_max,temp_min2,scaler2,scaler1)
# print(result[0][0][0])
# maximum=[]
# minimum=[]
# for i in range(30):
#     minimum.append(result[i][0][0][0])
#     maximum.append(result[i][1][0][0])
# plt.plot(maximum,label='max')
# plt.plot(minimum,label='min')
# plt.legend()
# plt.show()
    #反向解
        # tem_predicted=(predictions/scaler2)+temp_min2
        # real= (y_test1.reshape(-1,1)/scaler2)+y2.min(axis=0)
print('*'*100)

    # print(X_test)
    # print(RMSE)
    # print(len(tem_predicted))
    #求RMSE:
    # mse=mean_squared_error(real,tem_predicted)
    # rmse=np.sqrt(mse)
    # print(rmse,len(rmse),max(rmse),average(rmse))
    # pd.DataFrame(tem_predicted).to_csv('{}prediction.csv'.format(city),header=['min'])
    # 画损失function
    # plt.figure(figsize=(20,8),dpi=100)
    # plt.plot(history.history['loss'],label='Training LOSS')
    # plt.plot(history.history['val_loss'],label='VAL_LOSS')
    # plt.legend()
    # plt.show()

    #画预测/真实值对比function:
    # plt.figure(figsize=(20,8),dpi=100)
    # plt.plot(real,label='real')
    # plt.plot(tem_predicted,label='prediction')
    # plt.legend()
    # plt.show()

# predictions=(model.predict(X_test[-30:])/scaler1)+y1.min(axis=0)
# print(predictions)
