import re
from pymysql import *
con=connect(host='localhost',user='root',password='123456',database='tianqi',port=3306)
cursor=con.cursor()  #创建游标

def query(sql,params,type='no_select'):
    params=tuple(params)
    cursor.execute(sql,params)   #执行
    if type!='no-select':
        data_list=cursor.fetchall()
        con.commit()
        return data_list
    else:
        con.commit()
        return 'Conduction Success'

def save_to_sql(datas,city):
    p1=list(datas['最高温度'].str[:-1])
    p2=list(datas['最低温度'].str[:-1])
    for i in range(len(datas)):
        date=datas.iloc[i,0]
        max_temp=p1[i]
        min_temp=p2[i]
        weh=datas.iloc[i,3]
        wind=datas.iloc[i,4]
        query('insert into weather_record(city,record_date,max_temp,min_temp,weh,wind) values (%s,%s,%s,%s,%s,%s)',[city,date,max_temp,min_temp,weh,wind])
    con.commit()
    print('success')

def save_to_sql1(datas,city):
    p1=list(datas['最高温度'].str[:-1])
    p2=list(datas['最低温度'].str[:-1])
    for i in range(len(datas)):
        date=datas.iloc[i,0]
        max_temp=p1[i]
        min_temp=p2[i]
        weh=datas.iloc[i,3]
        wind=datas.iloc[i,4]
        query('insert into weatherdata(city,record_date,max_temp,min_temp,weh,wind) values (%s,%s,%s,%s,%s,%s)',[city,date,max_temp,min_temp,weh,wind])
    con.commit()
    print('success')

def save_to_sql2(datas, city):

    date = datas.iloc[0, 0]
    weh = datas.iloc[0, 1]
    wind = datas.iloc[0, 2]
    query('insert into future_info(city,record_date,weh,wind) values (%s,%s,%s,%s)',
          [city, date, weh, wind])
    con.commit()
    print('success')

def save_to_sql3(datas, city):

    date = datas.iloc[0, 0]
    weh = datas.iloc[0, 1]
    wind = datas.iloc[0, 2]
    query('insert into future_prediction(city,record_date,weh,wind) values (%s,%s,%s,%s)',
          [city, date, weh, wind])
    con.commit()
    print('success')

def save_to_sql4(identity, temp1, temp2):

    query('update future_prediction set max_prediction=%s, min_prediction=%s where id=%s',
          [temp1,temp2,identity])
    con.commit()
    print('success')
