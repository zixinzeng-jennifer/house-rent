import requests
from lxml import etree
import time
from fake_useragent import UserAgent
import xlrd
import xlwt

dictionary = ['link','租房网站名称','小区','详细地址','城市','区','价格（每月，单位为元）','	户型（室）','户型（卫）','户型（厅）','面积（单位：平方米）','朝向（东、南、西、北）','所属楼层（高、中、低）','总楼层（数字）','是否有阳台','信息发布人类型（经纪人 or 个人）','是否有床','是否有衣柜','是否有沙发','是否有电视','是否有冰箱','是否有洗衣机','是否有空调','是否有热水器','是否有宽带','是否有燃气','是否有暖气','最近公交距离','最近地铁距离','最近商场距离','最近医院距离','最近学校距离','周边公交个数','周边地铁个数','周边商场个数','周边医院个数','周边学校个数']
bj_area = ['朝阳','海淀','昌平','丰台','大兴','通州','房山','顺义','西城','东城','石景山','怀柔','门头沟']
device_list = ['阳台','信息发布人','床','衣柜','沙发','电视','冰箱','洗衣机','是否有空调','热水器','宽带','可做饭','暖气']
url_file = open('url_.txt')
url_str = url_file.read()
url_list = url_str.split('\n')
#print(url_list)
url_num = len(url_list)

workbook = xlwt.Workbook()
worksheet = workbook.add_sheet('sheet1')
for j in range(0,len(dictionary)):
    worksheet.write(0,j,dictionary[j])


i=0
url_end='?rfss=1-ccb332b5bf0bca0286-76'    #rfss会更新，需要调整
for url_head in url_list:
    url = url_head+url_end
    headers = {'User-Agent':'UserAgent().random'}
    time.sleep(1)
    html = requests.get(url,headers=headers)
    selector = etree.HTML(html.text)
    area = selector.xpath('//*[@id="agantzfxq_C01_03"]/text()')
    print(area)
    if area==[]:
        continue
    #print(area[0:-2])
    elif area[0][0:-2] in bj_area:
        xiaoqu = selector.xpath('//*[@id="agantzfxq_C02_07"]/text()')
        if xiaoqu==[]:
            continue
        addresses = selector.xpath('/html/body/div[6]/div[1]/div[4]/div[5]/div')
        if addresses==[]:
            continue
        i = i+1
        worksheet.write(i,0,url_head)
        worksheet.write(i,1,'房天下')
        worksheet.write(i,5,area[0][0:-2])
        worksheet.write(i,2,xiaoqu[0])
        address = addresses[-1].xpath('div[2]/a/text()')[0]
        #print(address)
        worksheet.write(i,3,address)
        worksheet.write(i,4,'北京')
        price = selector.xpath('/html/body/div[6]/div[1]/div[4]/div[2]/div[1]/i/text()')[0]
        worksheet.write(i,6,price)
        huxing = selector.xpath('/html/body/div[6]/div[1]/div[4]/div[3]/div[2]/div[1]/text()')[0]
        worksheet.write(i,7,huxing[0])
        worksheet.write(i,8,huxing[4])
        worksheet.write(i,9,huxing[2])
        square = selector.xpath('/html/body/div[6]/div[1]/div[4]/div[3]/div[3]/div[1]/text()')[0]
        worksheet.write(i,10,square[0:-2])
        direction = selector.xpath('/html/body/div[6]/div[1]/div[4]/div[4]/div[1]/div[1]/text()')[0]
        worksheet.write(i,11,direction)
        floor = selector.xpath('/html/body/div[6]/div[1]/div[4]/div[4]/div[2]/div[1]/text()')[0]
        worksheet.write(i,12,floor)
        floors = selector.xpath('/html/body/div[6]/div[1]/div[4]/div[4]/div[2]/div[2]/text()')[0]
        worksheet.write(i,13,floors[4:-2])
        devices = selector.xpath('/html/body/div[6]/div[1]/div[3]/div[2]/div[2]/div[2]/ul/li')
        for device_li in devices:
            device = device_li.xpath('text()')[0]
            #print(device)
            if device in device_list:
                device_index =device_list.index(device)
                #print(device_index)
                worksheet.write(i,14+device_index,device)
        workbook.save('data_bj.xls')
    print(i)

#workbook.save('data_bj.xls')
