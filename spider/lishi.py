import requests
import re
import time as delay
from bs4 import BeautifulSoup
import pandas as pd
from setuptools.package_index import user_agent
from selenium.webdriver.chrome.options import Options
from userUtils.query import save_to_sql
import json
from selenium import webdriver
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
    date_box=[]
    max_temp=[]
    min_temp=[]
    weh=[]
    wind=[]
    # 使用 BeautifulSoup 解析网页
    bs=BeautifulSoup(html,'html.parser')
    # 找到包含数据的标签
    data=bs.find_all(class_='thrui')
    # 使用正则表达式提取数据
    date = re.compile('class="th200">(.*?)</div>')
    tem = re.compile('class="th140">(.*?)</div>')
    time = re.findall(date, str(data))
    for item in time:
        # 不提取星期数据
        date_box.append(item[:10])
    temp = re.findall(tem, str(data))
    for i in range(len(temp) // 4):
        max_temp.append(temp[i * 4 + 0])
        min_temp.append(temp[i * 4 + 1])
        weh.append(temp[i * 4 + 2])
        wind.append(temp[i * 4 + 3])

    # 将数据转换为数据框，不添加星期列
    datas=pd.DataFrame({'日期':date_box,'最高温度':max_temp,'最低温度':min_temp,'天气':weh,'风向':wind })
    return datas


# def crawl_weather(city, code, time, db, cursor):
def crawl_weather(city, code, time):
    # 根据城市的编码和时间范围，生成网页的 url
    url = "http://lishi.tianqi.com/{}/{}.html".format(code, time)
    #下面是尝试直接使用请求获取页面信息，但是网站做了反爬虫，所以读不完数据，因此注释掉了
    # url2="https://lishi.tianqi.com/monthdata/{}/{}.html".format(code, time)
    # 定义请求头，模拟浏览器访问
    # headers2={#第二请求头，来获取余下月份的信息
    # 'accept':'application/json, text/javascript, */*;',
    # 'accept-encoding':'gzip, deflate, br, zstd',
    # 'accept-language':'zh-CN,zh;q=0.9',
    # 'content-length':'55',
    # 'content-type':'application/x-www-form-urlencoded; charset=UTF-8',
    # 'cookie':'UserId=17430598577415539; Hm_lvt_7c50c7060f1f743bccf8c150a646e90a=1743059858; HMACCOUNT=7B69277AB455F749; Hm_lvt_30606b57e40fddacb2c26d2b789efbcb=1743059875; UserId=17430628048595295; Hm_lvt_ab6a683aa97a52202eab5b3a9042a8d2=1743062805; HMACCOUNT=7B69277AB455F749; cityPy=guangzhou; cityPy_expire=1743670602; Hm_lpvt_ab6a683aa97a52202eab5b3a9042a8d2=1743065942; Hm_lpvt_30606b57e40fddacb2c26d2b789efbcb=1743067090; Hm_lpvt_7c50c7060f1f743bccf8c150a646e90a=1743073159'
    # ,'origin':'https://lishi.tianqi.com'
    # ,'priority':'u=1,i',
    # 'referer':'https://lishi.tianqi.com/{}/{}.html'.format(code, time),
    # 'sec-ch-ua':'"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
    # 'sec-ch-ua-mobile':'?0',
    # 'sec-ch-ua-platform':"Windows",
    # 'sec-fetch-dest':'empty',
    # 'sec-fetch-mode':'cors',
    # 'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
    # 'x-requested-with':'XMLHttpRequest'}
    # headers = {  # 请求标头，通过模拟请求标头以此实现仿人类登录进入网站并获取信息
    #     'User-Agent':' Mozilla/5.0 (Windows NT 10.0; WOW64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5756.197 Safari/537.36',
        # 'Cookie': 'lianjia_uuid=9d3277d3-58e4-440e-bade-5069cb5203a4;'
        #           'UM_distinctid=16ba37f7160390-05f17711c11c3e-454c0b2b-100200-16ba37f716618b;'
        #           ' _smt_uid=5d176c66.5119839a;'
        #           'sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%2216ba37f7a942a6-0671dfdde0398a-454c0b2b-1049088-16ba37f7a95409%22%2C%22%24device_id%22%3A%2216ba37f7a942a6-0671dfdde0398a-454c0b2b-1049088-16ba37f7a95409%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_referrer%22%3A%22%22%2C%22%24latest_referrer_host%22%3A%22%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%7D%7D; '
        #           '_ga=GA1.2.1772719071.1561816174; '
        #           'Hm_lvt_9152f8221cb6243a53c83b956842be8a=1561822858;'
        #           '_jzqa=1.2532744094467475000.1561816167.1561822858.1561870561.3;'
        #           'CNZZDATA1253477573=987273979-1561811144-%7C1561865554;'
        #           'CNZZDATA1254525948=879163647-1561815364-%7C1561869382;'
        #           'CNZZDATA1255633284=1986996647-1561812900-%7C1561866923;'
        #           'CNZZDATA1255604082=891570058-1561813905-%7C1561866148;'
        #           '_qzja=1.1577983579.1561816168942.1561822857520.1561870561449.1561870561449.1561870847908.0.0.0.7.3;'
        #           'select_city=110000; lianjia_ssid=4e1fa281-1ebf-e1c1-ac56-32b3ec83f7ca;'
        #           'srcid=eyJ0Ijoie1wiZGF0YVwiOlwiMzQ2MDU5ZTQ0OWY4N2RiOTE4NjQ5YmQ0ZGRlMDAyZmFhO'
        #           'DZmNjI1ZDQyNWU0OGQ3MjE3Yzk5NzFiYTY4ODM4ZThiZDNhZjliNGU4ODM4M2M3ODZhNDNiNjM1NzMzNjQ4'
        #           'ODY3MWVhMWFmNzFjMDVmMDY4NWMyMTM3MjIxYjBmYzhkYWE1MzIyNzFlOGMyOWFiYmQwZjBjYjcyNmI'
        #           'wOWEwYTNlMTY2MDI1NjkyOTBkNjQ1ZDkwNGM5ZDhkYTIyODU0ZmQzZjhjODhlNGQ1NGRkZTA0ZTBlZDFiN'
        #           'mIxOTE2YmU1NTIxNzhhMGQ3Yzk0ZjQ4NDBlZWI0YjlhYzFiYmJlZjJlNDQ5MDdlNzcxMzAwMmM1ODBlZDJkNm'
        #           'IwZmY0NDAwYmQxNjNjZDlhNmJkNDk3NGMzOTQxNTdkYjZlMjJkYjAxYjIzNjdmYzhiNzMxZDA1MGJlNjBmNzQ'
        #           'xMTZjNDIzNFwiLFwia2V5X2lkXCI6XCIxXCIsXCJzaWduXCI6XCIzMGJlNDJiN1wifSIsInIiOiJodHRwczovL2'
        #           'JqLmxpYW5qaWEuY29tL3p1ZmFuZy9yY28zMS8iLCJvcyI6IndlYiIsInYiOiIwLjEifQ=='
    #}



    try:
        driver = webdriver.Chrome()
        driver.get(url)
        delay.sleep(2)
        result=driver.find_element(By.CSS_SELECTOR,'body > div.main.clearfix > div.main_left.inleft > div.tian_three > ul > div')
        ActionChains(driver).click(result).perform()
        delay.sleep(2)
        html = driver.page_source
        # print('xxx')
        datas = parse_page(html)
        print(datas)
        save_to_sql(datas, city)
        print("成功爬取 {} 的 {} 的历史天气数据".format(city, time))
    except Exception as e:
        print("爬取 {} 的 {} 历史天气数据失败：{}".format(city, time, e))


with open('city.json', 'r', encoding='utf-8') as f:
    city_dict = json.load(f) # 将json文件转换为字典


time_list = []
ylist=['2015','2016','2017','2018','2019','2020','2021']
for i in ylist:
    for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
        time_list.append(i+j)
# time_list = ['202101','202102','202103','202204','202205','202206','202207','202208','202209','202210','202211','202212',
#     '202201','202202','202203','202204','202205','202206','202207','202208','202209','202210','202211','202212',
#              '202301','202302','202303','202304','202305','202306','202307','202308','202309','202310','202311',"202312",
#              "202401","202402","202403","202404","202405","202406","202407","202408","202409","202410","202411","202412",
#              "202501","202502","202503"]



# 遍历城市字典和时间列表，调用爬取函数
for city, code in city_dict.items():
    for time in time_list:

        crawl_weather(city, code, time)  # 传递 cursor 参数

        delay.sleep(6)

