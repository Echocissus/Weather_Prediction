import re
from pymysql import *
con=connect(host='localhost',user='root',password='123456',database='tianqi',port=3306)
cursor=con.cursor()  #创建游标
