from unittest.mock import inplace

import pandas as pd
import numpy as np
from pandas import isnull

from userUtils import query
import sqlalchemy
from sqlalchemy import create_engine
from pymysql import *
import json

def fetch_and_clean_gz():
    DB_USER='root'
    DB_PASS='123456'
    DB_HOST='localhost'
    DATABASE='tianqi'
    DB_PORT=3306
    connect_info = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(DB_USER, DB_PASS, DB_HOST, DB_PORT, DATABASE)  #1
    engine = create_engine(connect_info)
    query1='select distinct * from weather_record where city="广州" order by record_date'
    original_data=pd.read_sql(query1,engine)

    #求风力
    def findwind(i,j):
        try:
            return eval(str(mid[i].iloc[j,5])[-2])
        except:
            return 0




    #按城市分组
    original_weather_list = []
    original_weather_list.append(original_data.loc[original_data["city"]=='广州'])
    date_list=pd.DataFrame(list(pd.date_range('2015-01-01','2025-03-29',freq='D')),index=list(pd.date_range('2015-01-01','2025-03-29',freq='D')))
    # print(original_weather_list,type(original_weather_list[0]))
    # print(date_list)



    #解析增加变量
    mid=list(range(1))
    for i in range(1):
        mid[i]=original_weather_list[i].set_index('record_date',drop=True)
        # print(mid[i])
        sunny=pd.Series(list(np.zeros(len(mid[i]))))
        # print(sunny.info())
        cloudy=pd.Series(list(np.zeros(len(mid[i]))))
        rainy=pd.Series(list(np.zeros(len(mid[i]))))
        snowy=pd.Series(list(np.zeros(len(mid[i]))))
        overcast=pd.Series(list(np.zeros(len(mid[i]))))
        eastwind = pd.Series(list(np.zeros(len(mid[i]))))
        westwind = pd.Series(list(np.zeros(len(mid[i]))))
        northwind = pd.Series(list(np.zeros(len(mid[i]))))
        southwind = pd.Series(list(np.zeros(len(mid[i]))))
        wind_level= pd.Series(list(np.zeros(len(mid[i]))))
        for j in range(len(mid[i])):
            sunny[j]=int(bool(str(mid[i].iloc[j,4]).find('晴')!=-1))
            overcast[j]=int(bool(str(mid[i].iloc[j,4]).find('阴')!=-1))
            rainy[j]=int(bool(str(mid[i].iloc[j,4]).find('雨')!=-1))
            cloudy[j]=int(bool(str(mid[i].iloc[j,4]).find('多云')!=-1))
            snowy[j]=int(bool(str(mid[i].iloc[j,4]).find('雪')!=-1))
            eastwind[j]=int(bool(str(mid[i].iloc[j,5]).find('东')!=-1))
            westwind[j] = int(bool(str(mid[i].iloc[j, 5]).find('西') !=-1))
            northwind[j] = int(bool(str(mid[i].iloc[j, 5]).find('北') !=-1))
            southwind[j] = int(bool(str(mid[i].iloc[j, 5]).find('南') !=-1))
            wind_level[j]=findwind(i,j)
        sunny=pd.DataFrame(sunny.values,index=mid[i].index).astype('bool')
        overcast=pd.DataFrame(overcast.values,index=mid[i].index).astype('bool')
        rainy=pd.DataFrame(rainy.values,index=mid[i].index).astype('bool')
        cloudy=pd.DataFrame(cloudy.values,index=mid[i].index).astype('bool')
        snowy=pd.DataFrame(snowy.values,index=mid[i].index).astype('bool')
        eastwind = pd.DataFrame(eastwind.values, index=mid[i].index).astype('bool')
        westwind = pd.DataFrame(westwind.values, index=mid[i].index).astype('bool')
        northwind = pd.DataFrame(northwind.values, index=mid[i].index).astype('bool')
        southwind = pd.DataFrame(southwind.values, index=mid[i].index).astype('bool')
        wind_level = pd.DataFrame(wind_level.values, index=mid[i].index).astype('bool')
        mid[i]=  mid[i].join(sunny)
        mid[i]=mid[i].rename(columns={0:'sunny'})
        mid[i] = mid[i].join(overcast)
        mid[i].rename(columns={0:'overcast'}, inplace=True)
        mid[i] = mid[i].join(rainy)
        mid[i].rename(columns={0:'rainy'}, inplace=True)
        mid[i] = mid[i].join(cloudy)
        mid[i].rename(columns={0:'cloudy'}, inplace=True)
        mid[i] = mid[i].join(snowy)
        mid[i].rename(columns={0:'snowy'}, inplace=True)
        mid[i] = mid[i].join(eastwind)
        mid[i] = mid[i].rename(columns={0: 'EW'})
        mid[i] = mid[i].join(westwind)
        mid[i] = mid[i].rename(columns={0: 'WW'})
        mid[i] = mid[i].join(northwind)
        mid[i] = mid[i].rename(columns={0: 'NW'})
        mid[i] = mid[i].join(southwind)
        mid[i] = mid[i].rename(columns={0: 'SW'})
        mid[i] = mid[i].join(wind_level)
        mid[i] = mid[i].rename(columns={0: 'wind_level'})
        mid[i] = date_list.join(mid[i])
    #     print(mid[i].info())
    # print(mid[16])

    for i in range(1):
        for j in range(len(mid[i])):
            if isnull(mid[i].iloc[j, 3]) == 1:
                mid[i].iloc[j, 0] = mid[i].iloc[j - 1, 0] + pd.Timedelta(days=1)
                for k in range(len(mid[i].columns)):
                    mid[i].iloc[j, k] = mid[i].iloc[j - 1, k]


    y=[]
    m=[]
    d=[]

    for i in range(1):
        try:
            mid[i].rename(columns={0: 'date'},inplace=True)
            y.append(list(pd.to_datetime(mid[i]['date'], format='ISO8601').dt.year))
            m.append(list(pd.to_datetime(mid[i]['date'], format='ISO8601').dt.month))
            d.append(list(pd.to_datetime(mid[i]['date'], format='ISO8601').dt.day))
            year=pd.DataFrame(y[i], index=mid[i].index).astype('float32')
            month=pd.DataFrame(m[i], index=mid[i].index).astype('float32')
            day=pd.DataFrame(d[i], index=mid[i].index).astype('float32')
            mid[i]=mid[i].join(year)
            mid[i].rename(columns={0: 'year'},inplace=True)
            mid[i] = mid[i].join(month)
            mid[i].rename(columns={0: 'month'},inplace=True)
            mid[i] = mid[i].join(day)
            mid[i].rename(columns={0: 'day'},inplace=True)
            mid[i]=mid[i].drop(['date','id'],axis=1)
        except:
            mid[i]=0

    # mid[26].rename(columns={0: 'date'}, inplace=True)
    # y.append(list(pd.to_datetime(mid[26]['date'], format='ISO8601').dt.year))
    # m.append(list(pd.to_datetime(mid[26]['date'], format='ISO8601').dt.month))
    # d.append(list(pd.to_datetime(mid[26]['date'], format='ISO8601').dt.day))
    # year = pd.DataFrame(y[0], index=date_list.index).astype('float64')
    # month = pd.DataFrame(m[0], index=date_list.index).astype('float64')
    # day = pd.DataFrame(d[0], index=date_list.index).astype('float64')
    # mid[26] = mid[26].join(year)
    # mid[26].rename(columns={0: 'year'}, inplace=True)
    # mid[26] = mid[26].join(month)
    # mid[26].rename(columns={0: 'month'}, inplace=True)
    # mid[26] = mid[26].join(day)
    # mid[26].rename(columns={0: 'day'}, inplace=True)
    # mid[26] = mid[26].drop(['date', 'id'], axis=1)
    # print(mid[26])
    try:
        for city in range(1):
            if bool(type(mid[city]) == int):
                mid.pop(city)
            mid[city].to_csv('datas/{}data.csv'.format(mid[city].iloc[1,0]))
    except:
        return mid
a=fetch_and_clean_gz()


