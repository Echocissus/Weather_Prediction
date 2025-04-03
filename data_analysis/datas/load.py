import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import random
from sqlalchemy import create_engine
# from separate_training import predict
# from training import predict2
# import tensorflow as tf

def findwind(j):
    try:
        return eval(str(data.iloc[j, 5])[-2])
    except:
        return 0

DB_USER='root'
DB_PASS='123456'
DB_HOST='localhost'
DATABASE='tianqi'
DB_PORT=3306
connect_info = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(DB_USER, DB_PASS, DB_HOST, DB_PORT, DATABASE)  #1
engine = create_engine(connect_info)
query1='select distinct * from future_info where city="广州" order by record_date'
data=pd.read_sql(query1,engine)
data['max_temp']=[i for i in range(30)]
data['min_temp']=[i for i in range(30)]
sunny = pd.Series(list(np.zeros(30)))
# print(sunny.info())
cloudy = pd.Series(list(np.zeros(30)))
rainy = pd.Series(list(np.zeros(30)))
snowy = pd.Series(list(np.zeros(30)))
overcast = pd.Series(list(np.zeros(30)))
eastwind = pd.Series(list(np.zeros(30)))
westwind = pd.Series(list(np.zeros(30)))
northwind = pd.Series(list(np.zeros(30)))
southwind = pd.Series(list(np.zeros(30)))
wind_level = pd.Series(list(np.zeros(30)))
data=data[['record_date','city','max_temp','min_temp','weh','wind']].set_index(['record_date'],drop=True)
print(data)

for j in range(30):
    sunny[j] = int(bool(str(data.iloc[j, 3]).find('晴') != -1))
    overcast[j] = int(bool(str(data.iloc[j, 3]).find('阴') != -1))
    rainy[j] = int(bool(str(data.iloc[j, 3]).find('雨') != -1))
    cloudy[j] = int(bool(str(data.iloc[j, 3]).find('多云') != -1))
    snowy[j] = int(bool(str(data.iloc[j, 3]).find('雪') != -1))
    eastwind[j] = int(bool(str(data.iloc[j, 4]).find('东') != -1))
    westwind[j] = int(bool(str(data.iloc[j, 4]).find('西') != -1))
    northwind[j] = int(bool(str(data.iloc[j, 4]).find('北') != -1))
    southwind[j] = int(bool(str(data.iloc[j, 4]).find('南') != -1))
    wind_level[j] = findwind(j)
sunny=pd.DataFrame(sunny.values,index=data.index).astype('bool')
overcast=pd.DataFrame(overcast.values,index=data.index).astype('bool')
rainy=pd.DataFrame(rainy.values,index=data.index).astype('bool')
cloudy=pd.DataFrame(cloudy.values,index=data.index).astype('bool')
snowy=pd.DataFrame(snowy.values,index=data.index).astype('bool')
eastwind = pd.DataFrame(eastwind.values, index=data.index).astype('bool')
westwind = pd.DataFrame(westwind.values, index=data.index).astype('bool')
northwind = pd.DataFrame(northwind.values, index=data.index).astype('bool')
southwind = pd.DataFrame(southwind.values, index=data.index).astype('bool')
wind_level = pd.DataFrame(wind_level.values, index=data.index).astype('bool')
data=  data.join(sunny)
data=data.rename(columns={0:'sunny'})
data = data.join(overcast)
data.rename(columns={0:'overcast'}, inplace=True)
data = data.join(rainy)
data.rename(columns={0:'rainy'}, inplace=True)
data = data.join(cloudy)
data.rename(columns={0:'cloudy'}, inplace=True)
data = data.join(snowy)
data.rename(columns={0:'snowy'}, inplace=True)
data = data.join(eastwind)
data = data.rename(columns={0: 'EW'})
data = data.join(westwind)
data = data.rename(columns={0: 'WW'})
data = data.join(northwind)
data = data.rename(columns={0: 'NW'})
data = data.join(southwind)
data = data.rename(columns={0: 'SW'})
data = data.join(wind_level)
data = data.rename(columns={0: 'wind_level'})



c=pd.read_csv('广州data.csv')
c=c.set_index(['Unnamed: 0'],drop=True)

new_data=pd.concat([c,data],axis=0)
city=new_data.iloc[0,0]
new_data.to_csv('广州data_for_prediction.csv')




# predict(city)
# predict2(city)
# a=(pd.read_csv('广州data.csv')['max_temp']+pd.read_csv('广州data.csv')['max_temp'])/2
# a=a[-30:].values
# b=pd.read_csv('广州final_prediction.csv')['avg']
# b=b[-30:].values
# new_data.to_csv('广州data_for_prediction')
# c=mean_squared_error(a,b)
# rmse=np.sqrt(c)
# print(rmse)
# plt.plot(a,label='accurate')
# plt.plot(b,label='predicted')
# plt.show()

# c=a['max_temp'][-30:]
# d=a['min_temp'][-30:]
# e=(c+d)/2
# f=list(e.values)
# plt.plot(b)
# plt.plot(f)
# plt.show()
# # model=tf.keras.models.load_model('guangzhoumintemp_lstm.h5')
# prediction=model.predict(X_end)
# pre_temp=