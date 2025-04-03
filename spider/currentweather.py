import requests
import re
import time as delay
from bs4 import BeautifulSoup
import pandas as pd
from setuptools.package_index import user_agent
from selenium.webdriver.chrome.options import Options
from userUtils.query import save_to_sql1
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
import datetime as dt

current_month=a=str(dt.datetime.now())[:7].replace('-','')
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
    date_box = []
    max_temp = []
    min_temp = []
    weh = []
    wind = []
    # 使用 BeautifulSoup 解析网页
    bs = BeautifulSoup(html, 'html.parser')
    # 找到包含数据的标签
    data = bs.find_all(class_='thrui')
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
    datas = pd.DataFrame({'日期': date_box, '最高温度': max_temp, '最低温度': min_temp, '天气': weh, '风向': wind})
    return datas


def crawl_current_weather(city, code):

    url = "http://lishi.tianqi.com/{}/{}.html".format(code, current_month)

    try:
        driver = webdriver.Chrome()
        driver.get(url[:-5])
        delay.sleep(2)
        result = driver.find_element(By.CSS_SELECTOR,
                                     'body > div.main.clearfix > div.main_left.inleft > div.tian_three > ul > div')
        ActionChains(driver).click(result).perform()
        delay.sleep(2)
        html = driver.page_source
        # print('xxx')
        datas = parse_page(html)
        print(datas)
        save_to_sql1(datas, city)
        print("成功爬取 {} 的 {} 的历史天气数据".format(city, current_month))
    except Exception as e:
        print("爬取 {} 的 {} 历史天气数据失败：{}".format(city, current_month, e))


with open('city.json', 'r', encoding='utf-8') as f:
    city_dict = json.load(f)  # 将json文件转换为字典


# 遍历城市字典和时间列表，调用爬取函数
for city, code in city_dict.items():
    crawl_current_weather(city, code)  # 传递 cursor 参数
    delay.sleep(3)