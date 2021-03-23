import requests
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time


feature = ['link','webname','xiaoqu','address','city','district','price','shi','wei','ting','area','direction','floor','totalfloor','balcony','publisher','bed','closet','sofa','tv','refrigerator','washer','air-conditioning','water heater','broadband','gas','heater','underground_distance']
df = pd.read_csv('initial_data.csv', header=0, names=feature)
driver = webdriver.Chrome()


# 共有 9037 条
pointer = df.shape[0] # 行数
print(pointer)
for i in range(0,pointer):
    if pd.isnull(df.at[i,'price']):
        if i % 10 == 0:
            time.sleep(3)
        print('剩余   ', pointer - i, '   条,正在爬取第   ', i + 1, '   条，')
        url = df.loc[i, 'link']

        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        above = soup.find('div', class_='f-card f-er-card f-w1190 f-clear f-b30')
        try:
            info = above.find('div', class_='price-wrap')
        except AttributeError:
            print('页面不存在')
            df.loc[i,'price']='none'
            df.to_csv('initial_data.csv')
            continue

        price = info.find('span', class_='price').string

        info = above.find('ul', class_='er-list f-clear')
        li = info.find_all('li', class_='item f-fl')
        huxing = li[0].find('span', class_='content').string
        shi = huxing[0]
        ting = huxing[2]
        wei = huxing[4]
        area = li[1].find('span', class_='content').string[4:-1]
        direction = li[2].find('span', class_='content').string
        floor = li[3].find('span', class_='content').string[0:2]
        totalfloor = li[3].find('span', class_='content').string[5:-1]

        info = above.find('ul', class_='er-list-two f-clear')
        li = info.find_all('li', class_='er-item f-fl')
        xiaoqu = li[0].find('span', class_=None).string
        address = li[1].find('span', class_='content').string.lstrip()
        underground_distance = info.find('div', class_='subway-wrap').find('span', class_='content').string

        below = soup.find('div', class_='f-main f-w1190 f-clear')
        info = below.find('div', {'id': 'js-house-peizhi'})
        li = info.find_all('li')
        if li[0]['class'] == ['item']:
            tv = True
        else:
            tv = False

        if li[1]['class'] == ['item']:
            air_conditioning = True
        else:
            air_conditioning = False

        if li[2]['class'] == ['item']:
            water_heater = True
        else:
            water_heater = False

        if li[3]['class'] == ['item']:
            washer = True
        else:
            washer = False

        if li[4]['class'] == ['item']:
            refrigerator = True
        else:
            refrigerator = False

        if li[5]['class'] == ['item']:
            bed = True
        else:
            bed = False

        if li[6]['class'] == ['item']:
            sofa = True
        else:
            sofa = False

        if li[7]['class'] == ['item']:
            closet = True
        else:
            closet = False

        if li[8]['class'] == ['item']:
            heater = True
        else:
            heater = False

        if li[9]['class'] == ['item']:
            broadband = True
        else:
            broadband = False

        if li[10]['class'] == ['item']:
            gas = True
        else:
            gas = False

        if li[11]['class'] == ['item']:
            balcony = True
        else:
            balcony = False

        df.loc[i, 'xiaoqu'] = xiaoqu
        df.loc[i, 'address'] = address
        df.loc[i, 'price'] = price
        df.loc[i, 'shi'] = shi
        df.loc[i, 'wei'] = wei
        df.loc[i, 'ting'] = ting
        df.loc[i, 'area'] = area
        df.loc[i, 'direction'] = direction
        df.loc[i, 'floor'] = floor
        df.loc[i, 'totalfloor'] = totalfloor
        df.loc[i, 'balcony'] = balcony
        df.loc[i, 'bed'] = bed
        df.loc[i, 'closet'] = closet
        df.loc[i, 'sofa'] = sofa
        df.loc[i, 'tv'] = tv
        df.loc[i, 'refrigerator'] = refrigerator
        df.loc[i, 'washer'] = washer
        df.loc[i, 'air-conditioning'] = air_conditioning
        df.loc[i, 'water heater'] = water_heater
        df.loc[i, 'broadband'] = broadband
        df.loc[i, 'gas'] = gas
        df.loc[i, 'heater'] = heater
        df.loc[i, 'underground_distance'] = underground_distance

        # 一条存一下
        df.to_csv('initial_data.csv')

    else:
        print('第   ', i + 1, '   条已经存在，跳过，剩余   ', pointer - i - 1, '   条')













