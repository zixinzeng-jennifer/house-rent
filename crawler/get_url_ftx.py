import re  
from bs4 import BeautifulSoup  
import requests
from lxml import etree
import time
from fake_useragent import UserAgent

#citys = ['','sh.','sz.']
citys = ['']
url_head = 'https://'
url_middle = 'zu.fang.com/house/i3'
url_end='-n31/?rfss=2-8781a89e3d8e488514-11#'    #会更新，需要调整


for city in citys:
    url_list =''
    url_num =0
    for i in range(1,101):
        page = str(i)
        url = url_head+city+url_middle+page+url_end
        #print(url)
        #headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'}
        headers = {'User-Agent':'UserAgent().random'}
        #time.sleep(1)
        html = requests.get(url,headers=headers)
        selector = etree.HTML(html.text)
        infos = selector.xpath('//p[@class="title"]')
        for info in infos:
            url_info = info.xpath('a/@href')[0]
            url_num = url_num+1
            url_complete = 'https://'+city+'zu.fang.com'+url_info+'\n'
            url_list = url_list+url_complete
        #print(url_list)
        print(url_num)
    file_name = 'url_'+city+'.txt'
    file = open(file_name,"w", encoding='utf-8')
    file.write(url_list)
    file.close()
