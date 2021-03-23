import requests
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time



geo={}
geo['北京']=['朝阳','海淀','昌平','丰台','大兴','通州','房山','顺义','西城','东城','石景山','怀柔','门头沟']
geo['上海']=['浦东','闵行','宝山','嘉定', '徐汇', '静安', '普陀','杨浦','奉贤','黄浦','虹口','长宁']
geo['深圳']=['宝安','南山','龙华','福田','龙岗','罗湖','布吉','光明','盐田']
url_dict={}
url_dict['朝阳']='http://bj.ganji.com/chaoyang/zufang/'
url_dict['海淀']='http://bj.ganji.com/haidian/zufang/'
url_dict['昌平']='http://bj.ganji.com/changping/zufang/'
url_dict['丰台']='http://bj.ganji.com/fengtai/zufang/'
url_dict['大兴']='http://bj.ganji.com/daxing/zufang/'
url_dict['通州']='https://bj.ganji.com/tongzhouqu/zufang/'
url_dict['房山']='https://bj.ganji.com/fangshan/zufang/'
url_dict['顺义']='https://bj.ganji.com/shunyi/zufang/'
url_dict['西城']='https://bj.ganji.com/xicheng/zufang/'
url_dict['东城']='https://bj.ganji.com/dongcheng/zufang/'
url_dict['石景山']='https://bj.ganji.com/shijingshan/zufang/'
url_dict['怀柔']='https://bj.ganji.com/huairou/zufang/'
url_dict['门头沟']='https://bj.ganji.com/mentougou/zufang/'
url_dict['浦东']='https://sh.ganji.com/pudongxinqu/zufang/'
url_dict['闵行']='https://sh.ganji.com/minxing/zufang/'
url_dict['宝山']='https://sh.ganji.com/baoshan/zufang/'
url_dict['嘉定']='https://sh.ganji.com/jiading/zufang/'
url_dict['徐汇']='https://sh.ganji.com/xuhui/zufang/'
url_dict['静安']='https://sh.ganji.com/jingan/zufang/'
url_dict['普陀']='https://sh.ganji.com/putuo/zufang/'
url_dict['杨浦']='https://sh.ganji.com/yangpu/zufang/'
url_dict['奉贤']='https://sh.ganji.com/fengxiansh/zufang/'
url_dict['黄浦']='https://sh.ganji.com/huangpu/zufang/'
url_dict['虹口']='https://sh.ganji.com/hongkou/zufang/'
url_dict['长宁']='https://sh.ganji.com/changning/zufang/'
url_dict['宝安']='https://sz.ganji.com/baoan/zufang'
url_dict['南山']='https://sz.ganji.com/nanshan/zufang/'
url_dict['龙华']='https://sz.ganji.com/szlhxq/zufang/'
url_dict['福田']='https://sz.ganji.com/futian/zufang/'
url_dict['龙岗']='https://sz.ganji.com/longgang/zufang/'
url_dict['罗湖']='https://sz.ganji.com/luohu/zufang/'
url_dict['布吉']='https://sz.ganji.com/buji/zufang/'
url_dict['光明']='https://sz.ganji.com/guangmingxinqu/zufang/'
url_dict['盐田']='https://sz.ganji.com/yantian/zufang/'

memory = np.load('memory.npy', allow_pickle=True).item()
print('爬取情况',memory)



# memory={}
# for city in geo:
#     for district in geo[city]:#返回区的名称
#         memory[district] = False
# memory['朝阳']=True
# memory['海淀']=True
# memory['昌平']=True
# memory['丰台']=True
# memory['大兴']=True
# np.save('memory.npy', memory)

# USER_AGENT_LIST=[
#     "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36"
# ]
#
# urllist = []
# for line in open('urllist.txt','r'):
#     urllist.append(line)

feature = ['link','webname','xiaoqu','address','city','district','price','shi','wei','ting','area','direction','floor','totalfloor','balcony','publisher','bed','closet','sofa','tv','refrigerator','washer','broadband','gas','heater']
df = pd.read_csv('initial_data.csv', header=0, names=feature)


pointer = df.shape[0] # 行数
print('已有条数：',pointer)


# myheaders = {
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#     "Accept-Encoding": "br, gzip, deflate",
#     "Accept-Language": "zh-cn",
#     "Host": "httpbin.org",
#     "User-Agent": np.random.choice(USER_AGENT_LIST)
#   }


driver = webdriver.Chrome()
# driver.implicitly_wait(5)
# driver.get('http://bj.ganji.com/chaoyang/zufang/')
# html = driver.page_source
# soup = BeautifulSoup(html, 'html.parser')  # 指定Beautiful的解析器为“html.parser”



for city in geo:
    print(city)
    for district in geo[city]:#返回区的名称
        print(district)
        if memory[district]:
            print(district+'  已经爬取，跳过')
            continue

        base_url=url_dict[district]
        driver.get(base_url)
        html = driver.page_source
        for i in range(5):#爬取前5页的内容，每页约有110条数据
            print("第"+str(i+1)+"页")
            soup = BeautifulSoup(html, 'html.parser')  #指定Beautiful的解析器为“html.parser”
            # 指定属性查找标签 soup.find_all('tag', {'id': 'tag id', 'class': 'tag class'})
            # 这个循环的作用是找到本页面所有租房链接
            for item in soup.find_all('div',class_="f-list-item ershoufang-list"):
                publisher = item.find_all('span',{'class':'address-eara','target': None})
                # print(publisher)
                title = item.find('dd', class_="dd-item title")
                link = title.find('a')['href']
                if link[0:4] != "http":
                    link = "http:" + link

                if link in df['link'].tolist():
                    print(link+'重复')
                else:
                    # urllist.append(link)
                    # f.write(link + '\n')
                    df.loc[pointer,'link']= link
                    df.loc[pointer, 'webname'] = '赶集网'
                    df.loc[pointer, 'district'] = district
                    df.loc[pointer, 'city'] = city
                    if str(publisher)[28:33]=='来自经纪人':
                        df.loc[pointer,'publisher']='经纪人'
                        # print('jjr')
                    elif str(publisher)[28:33]=='来自个人房':
                        df.loc[pointer,'publisher']='个人'
                        # print('gr')
                    elif str(publisher)[28:32]=='安选企业':
                        df.loc[pointer,'publisher']='经纪人-安选企业'
                        # print('axqy')
                    else:
                        df.loc[pointer, 'publisher'] ='其他'+str(publisher)
                        # print('其他'+str(publisher))
                    pointer=pointer+1

            time.sleep(5)
            # 找到"下一页"链接
            pagediv = soup.find('div', {'class': "pageBox"})
            nextpage = pagediv.find('a', {'class': 'next'})
            pagelink = nextpage['href']
            driver.get(pagelink)
            html = driver.page_source
            print('目前数据条数：      ', pointer)


        # 保存爬取的数据，保存市区的字典
        df.to_csv('initial_data.csv')
        memory[district]=True
        np.save('memory.npy',memory)











# wd = webdriver.Chrome()
# wd.get("https://www.baidu.com")    # 打开百度浏览器
# wd.find_element_by_id("kw").send_keys("selenium")   # 定位输入框并输入关键字
# wd.find_element_by_id("su").click()   #点击[百度一下]搜索
# time.sleep(3)   #等待3秒
# wd.quit()   #关闭浏览器







