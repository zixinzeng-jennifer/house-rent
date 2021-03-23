import numpy as np
import pandas as pd


# def data_processing():

df = pd.read_csv('data_all.csv',header=0)
# print(df.head(5))

# 去除楼层异常数值（共867条）
df=df.dropna(subset=['总楼层（数字）'])
# 去除方向异常数值（共35条）
df=df[df['朝向（东、南、西、北）']!='暂无']
# 去除户型（室厅卫）异常值
df=df[df['户型（卫）']!='厅']
df=df[df['户型（厅）']!='室']
print('数据条数：',df.shape[0])
# 去掉周边医院个数为空的记录（46条））
df=df.dropna(subset=['周边医院个数'])


# 将最近学校/医院距离为空的填入-1
num_cols=['最近学校距离','最近医院距离']
for col in num_cols:
    df[col].fillna(-1, inplace=True)

df.info()


# 将非数值型特征进行独热编码
def direction_to_onehot(row):
    if row =='南':
        return 1
    elif row =='东西':
        return 2
    elif row =='西南':
        return 3
    elif row =='东':
        return 4
    elif row =='西':
        return 5
    elif row =='东北':
        return 6
    elif row =='西北':
        return 7
    elif row =='北':
        return 8
    elif row == '东西':
        return 4.5
    elif row == '南北':
        return 4.5
    # 不知道为什么会有空值nan？？？
    else:
        return 0

def floor_to_onehot(row):
    if row =='地下':
        return 1
    elif row =='低层':
        return 2
    elif row =='中层':
        return 3
    elif row =='高层':
        return 4

def publisher_to_onehot(row):
    if row=='经纪人':
        return 0
    elif row=='个人':
        return 1


df['朝向（东、南、西、北）']=df['朝向（东、南、西、北）'].apply(direction_to_onehot)
# # print(df['朝向（东、南、西、北）'].unique())
df['所属楼层（高、中、低）']=df['所属楼层（高、中、低）'].apply(floor_to_onehot)
# # print(df['所属楼层（高、中、低）'].unique())
df['信息发布人类型（经纪人 or 个人）']=df['信息发布人类型（经纪人 or 个人）'].apply(publisher_to_onehot)
df['户型（卫）']=[int(x) for x in df['户型（卫）']]
df['户型（厅）']=[int(x) for x in df['户型（厅）']]


feature=['价格（每月，单位为元）',"户型（室）",'户型（卫）','户型（厅）','面积（单位：平方米）','朝向（东、南、西、北）','所属楼层（高、中、低）','总楼层（数字）','是否有阳台','信息发布人类型（经纪人 or 个人）','是否有床','是否有衣柜','是否有沙发','是否有电视','是否有冰箱','是否有洗衣机','是否有空调','是否有热水器','是否有宽带','是否有燃气','是否有暖气','lng','lat','最近学校距离','周边学校个数','最近医院距离','周边医院个数']
for colu in feature:
    print(colu+'  取值：  ',df[colu].unique())

df.to_csv('processed_data.csv',index=False)





