import requests
import re
import time as delay
from bs4 import BeautifulSoup
import pandas as pd
from setuptools.package_index import user_agent
from selenium.webdriver.chrome.options import Options
from userUtils.query import save_to_sql2,save_to_sql3
import json
from selenium import webdriver
import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains

# 定义一个函数，用于获取网页的源代码
def get_page(url, headers):
    html = requests.get(url, headers=headers)
    # html = requests.get(url)
    if html.status_code == 200:
        html.encoding = html.apparent_encoding
        return html.text
    else:
        print("Failed to fetch page:", url)  # 添加调试语句
        return None


# 定义一个函数，用于解析网页中的数据，并返回一个数据框
def parse_page(html):
    # 创建空列表，用于存储数据
    # date_box=[]
    # max_temp=[]
    # min_temp=[]
    weh=[]
    wind=[]
    print(time)
    date=pd.to_datetime([time])
    # 使用 BeautifulSoup 解析网页
    bs=BeautifulSoup(html,'html.parser')
    # 找到包含数据的标签
    print(bs)
    data=bs.find_all(class_='tips_pretext')
    # 使用正则表达式提取数据
    # finddate=bs.findall(type="text/javascript")
    day = re.compile('datetimes(.*?);')
    # date=re.findall(day,str(finddate))
    weather = re.compile('class="gold">(.*?)</span>')
    windinfo = re.compile('class="gold">(.*?)</span>')
    # wea=bs.find_all(weather,str(data))
    win = re.findall(windinfo, str(data))
    print(date)
    wind.append(win)
    print(wind)
        # print(date_box,weh,wind )
    # temp = re.findall(tem, str(data))
    # for i in range(len(temp) // 4):
    #     # max_temp.append(temp[i * 4 + 0])
    #     # min_temp.append(temp[i * 4 + 1])
    #     weh.append(temp[i * 4 + 2])
    #     wind.append(temp[i * 4 + 3])

    # 将数据转换为数据框，不添加星期列
    datas=pd.DataFrame({'日期':date,'天气':wind[0][1],'风向':wind[0][3] })
    return datas


# def crawl_weather(city, code, time, db, cursor):
def crawl_weather(city, code, time):
    # 根据城市的编码和时间范围，生成网页的 url
    url = "https://www.tianqi.com/tianqi/{}/{}.html".format(code, time)
    #下面是尝试直接使用请求获取页面信息，但是网站做了反爬虫，所以读不完数据，因此注释掉了



    try:
        driver = webdriver.Chrome()
        driver.get(url)
        # delay.sleep(2)
        # result=driver.find_element(By.CSS_SELECTOR,'body > div.main.clearfix > div.main_left.inleft > div.tian_three > ul > div')
        # ActionChains(driver).click(result).perform()
        delay.sleep(2)
        html = driver.page_source
        # print('xxx')
        datas = parse_page(html)
        print(datas)
        # save_to_sql2(datas, city)
        save_to_sql3(datas,city)
        print("成功爬取 {} 的 {} 的历史天气数据".format(city, time))
    except Exception as e:
        print("爬取 {} 的 {} 历史天气数据失败：{}".format(city, time, e))


with open('city.json', 'r', encoding='utf-8') as f:
    city_dict = json.load(f) # 将json文件转换为字典


time_list = []
ylist=['2025']
dlist=['01','02','03','04','05','06','07','08','09']+[str(i) for i in range(10,31)]
for i in ylist:
    for j in ['04']:
        for k in dlist:
            time_list.append(i+j+k)
print(time_list)
# time_list = ['202101','202102','202103','202204','202205','202206','202207','202208','202209','202210','202211','202212',
#     '202201','202202','202203','202204','202205','202206','202207','202208','202209','202210','202211','202212',
#              '202301','202302','202303','202304','202305','202306','202307','202308','202309','202310','202311',"202312",
#              "202401","202402","202403","202404","202405","202406","202407","202408","202409","202410","202411","202412",
#              "202501","202502","202503"]



# 遍历城市字典和时间列表，调用爬取函数
for city, code in city_dict.items():
    for time in time_list:

        crawl_weather(city, code, time)  # 传递 cursor 参数

        delay.sleep(3)