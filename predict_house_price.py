from __future__ import division
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import LinearSVR,SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet,RidgeCV
from sklearn.ensemble import RandomForestRegressor,StackingRegressor,BaggingRegressor,VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,explained_variance_score
import xgboost as xgb
from pyecharts.charts import Geo
from pyecharts import options as opts
from pyecharts.globals import ThemeType


def read_data():
    df = pd.read_csv('./webdatamining/house/processed_data.csv', header=0)
    print(df.head())
    print(df.columns)
    # print(df.shape)
    # print(df.info())
    # print(df.describe())
    print("缺失值:\n",df.isnull().sum())
    return df

def EDA(df):

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    sns.set_theme(style="whitegrid",palette="flare",font='SimHei', font_scale=1.5)

    plt.figure(figsize=[9, 7])
    df['城市'].value_counts().plot.pie(autopct='%1.1f')
    plt.savefig("./webdatamining/house/pics/city_sample.png")
    plt.show()
    plt.figure(figsize=[9, 7])
    df['租房网站名称'].value_counts().plot.pie(autopct='%1.1f')
    plt.savefig("./webdatamining/house/pics/website_sample.png")
    plt.show()
    sns.kdeplot(data=df[df['价格']<60000], x="价格", hue="城市", multiple="stack",palette="Spectral")
    plt.xlabel("价格(<6w 元)")
    plt.subplots_adjust(bottom=0.4)
    plt.title("各城市房价分布")
    plt.savefig("./webdatamining/house/pics/price_dist.png")
    plt.show()
    sns.scatterplot(data=df[df['面积']<2000],x="面积",y="价格")
    plt.xlabel("面积(<2000平方米)")
    plt.ylabel("价格")
    plt.title("面积与价格的关系")
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig("./webdatamining/house/pics/price_area.png")
    plt.show()
    g = sns.FacetGrid(df[df['面积']<600], col="城市", height=3.5, aspect=.65)
    g.map(sns.kdeplot, "面积")
    g.set(xlim=(0, 500),  xticks=[100,300,500])
    plt.savefig("./webdatamining/house/pics/area_dist.png")
    plt.show()
    plt.figure(figsize=[100,300])
    df['所属楼层'].replace(1,'地下',inplace=True)
    df['所属楼层'].replace(2, '低层', inplace=True)
    df['所属楼层'].replace(3, '中层', inplace=True)
    df['所属楼层'].replace(4, '高层', inplace=True)
    df.rename(columns={"所属楼层": "楼层"}, inplace=True)
    with sns.plotting_context(font_scale=1):
        g = sns.FacetGrid(df[(df['楼层']!="地下")&(df['总楼层']<60)], col="城市", row="楼层",despine=False,margin_titles=True,row_order=["低层","中层","高层"])
        g.map_dataframe(sns.histplot, x="总楼层",binwidth=5)
        g.set_axis_labels("总楼层", "Count")
        plt.savefig("./webdatamining/house/pics/floors.png")
        plt.show()
    X=pd.DataFrame(df[["是否有阳台","是否有床","是否有衣柜","是否有沙发","是否有电视","是否有冰箱","是否有洗衣机","是否有空调","是否有热水器","是否有宽带","是否有暖气","是否有燃气"]])
    X.drop_duplicates(inplace=True)
    print(X.head())
    g = sns.clustermap(X,col_cluster=True,row_cluster=False,cmap="mako", vmin=0, vmax=2)
    plt.savefig("./webdatamining/house/pics/infra.png")
    plt.show()

    #scaler=StandardScaler()
    df_shanghai=df[df["城市"]=="上海"]
    print(len(df_shanghai))
    df_house_coord_sh = {i: [df_shanghai.iloc[i]['lng'], df_shanghai.iloc[i]['lat']] for i in range(len(df_shanghai))}
    data_pair_sh = [(i, df_shanghai.iloc[i]['价格']/10000) for i in range(len(df_shanghai))]
    geo = Geo(init_opts=opts.InitOpts(theme=ThemeType.DARK))
    geo.add_schema(maptype='上海')
    for key, value in df_house_coord_sh.items():
        geo.add_coordinate(key, value[0], value[1])
    geo.add('', data_pair_sh, symbol_size=3, itemstyle_opts=opts.ItemStyleOpts(color="blue"))
    # 设置样式
    geo.set_series_opts(label_opts=opts.LabelOpts(is_show=True), type='heatmap')
    #  is_piecewise 是否自定义分段， 变为true 才能生效
    geo.set_global_opts(visualmap_opts=opts.VisualMapOpts(), title_opts=opts.TitleOpts(title="上海房价特点"))
    geo.render('./webdatamining/house/pics/shanghai_heat.html')

    df_beijing = df[df["城市"] == "北京"]
    print(len(df_beijing))
    df_house_coord_bj = {i: [df_beijing.iloc[i]['lng'], df_beijing.iloc[i]['lat']] for i in range(len(df_beijing))}
    data_pair_bj = [(i, df_beijing.iloc[i]['价格']/10000) for i in range(len(df_beijing))]
    geo = Geo(init_opts=opts.InitOpts(theme=ThemeType.DARK))
    geo.add_schema(maptype='北京')
    print(df_beijing.head())
    for key, value in df_house_coord_bj.items():
        geo.add_coordinate(key, value[0], value[1])
    geo.add('', data_pair_bj, symbol_size=3, itemstyle_opts=opts.ItemStyleOpts(color="blue"))
    # 设置样式
    geo.set_series_opts(label_opts=opts.LabelOpts(is_show=True), type='heatmap')
    #  is_piecewise 是否自定义分段， 变为true 才能生效
    geo.set_global_opts(visualmap_opts=opts.VisualMapOpts(), title_opts=opts.TitleOpts(title="北京房价特点"))
    geo.render('./webdatamining/house/pics/beijing_heat.html')

    df_shenzhen= df[df["城市"] == "深圳"]
    print(len(df_shenzhen))
    df_house_coord_sz = {i: [df_shenzhen.iloc[i]['lng'], df_shenzhen.iloc[i]['lat']] for i in range(len(df_shenzhen))}
    data_pair_sz = [(i, df_shenzhen.iloc[i]['价格']/10000) for i in range(len(df_shenzhen))]
    geo = Geo(init_opts=opts.InitOpts(theme=ThemeType.DARK))
    geo.add_schema(maptype='深圳')
    for key, value in df_house_coord_sz.items():
        geo.add_coordinate(key, value[0], value[1])
    geo.add('', data_pair_sz, symbol_size=3, itemstyle_opts=opts.ItemStyleOpts(color="blue"))
    # 设置样式
    geo.set_series_opts(label_opts=opts.LabelOpts(is_show=True), type='heatmap')
    #  is_piecewise 是否自定义分段， 变为true 才能生效
    geo.set_global_opts(visualmap_opts=opts.VisualMapOpts(), title_opts=opts.TitleOpts(title="深圳房价特点"))
    geo.render('./webdatamining/house/pics/shenzhen_heat.html')





def fit_SVMRegressor(df,flag_kernel=False,flag_search_param=False,flag_model_cross_valid=False,flag_get_feature_importance=True):
    if(flag_kernel==False and flag_search_param==False and flag_model_cross_valid==False and flag_get_feature_importance==False):
        print("Warning:all_flags are false")
    print('-'*70)
    print("支持向量机回归")
    #使用的特征
    predict_feature = ["室", '卫', '厅', '面积', '朝向', '所属楼层', '总楼层', '是否有阳台',
                       '信息发布人类型', '是否有床', '是否有衣柜', '是否有沙发', '是否有电视', '是否有冰箱', '是否有洗衣机', '是否有空调', '是否有热水器',
                       '是否有宽带', '是否有燃气', '是否有暖气', 'lng', 'lat', '最近学校距离', '周边学校个数', '最近医院距离', '周边医院个数']
    district = df['区'].tolist()
    input = df[predict_feature].values.tolist()
    output = df['价格'].tolist()
    #分割数据集
    train_data, test_data, train_target, test_target = train_test_split(input, output, test_size=0.25,stratify=district,random_state=42)
    if flag_kernel:
        print("选择合适的核函数")
        # ① 线性核函数
        model = LinearSVR(C=1,max_iter=5000)
        model.fit(train_data, train_target)
        y_pred = model.predict(test_data)
        print("线性核函数：(linear kernel)")
        basic_model_evaluation(model,train_data,train_target,test_data,test_target,y_pred)
        # ② 高斯核函数
        model = SVR(kernel='rbf', C=1, gamma=0.1, coef0=0.1)
        model.fit(train_data, train_target)
        y_pred = model.predict(test_data)
        basic_model_evaluation(model, train_data, train_target, test_data, test_target, y_pred)
        print("高斯核函数：(rbf kernel)")
        # ③ sigmoid核函数
        print("sigmoid 核函数：(sigmoid kernel)")
        model = SVR(kernel='sigmoid', C=1)
        model.fit(train_data, train_target)
        y_pred = model.predict(test_data)
        basic_model_evaluation(model, train_data, train_target, test_data, test_target, y_pred)
        # ④ 多项式核函数
        print("多项式核函数：(poly kernel)")
        model = SVR(kernel='poly', C=1)
        model.fit(train_data, train_target)
        y_pred = model.predict(test_data)
        basic_model_evaluation(model, train_data, train_target, test_data, test_target, y_pred)
    if flag_search_param:
        print("GridSearch:参数调优")
        model = GridSearchCV(LinearSVR(), param_grid={'C': [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5]}, cv=5)
        model.fit(train_data, train_target)
        print("best_param:", model.best_params_)
        print("best_score:", model.best_score_)
        model = GridSearchCV(SVR(),param_grid={'kernel':['poly','sigmoid','rbf'],'C': [0.1,0.5,1,5,10],'gamma': ["auto","scale"]},cv=5)
        model.fit(train_data, train_target)
        print("best_param:", model.best_params_)
        print("best_score:", model.best_score_)
    if flag_model_cross_valid:#使用Linear Kernel
        X = df[predict_feature]
        y = df['价格']
        # 训练回归模型
        n_folds = 5  # 设置交叉检验的次数
        model_1 = LinearSVR(C=0.1)  # 线性核函数
        model_2 = LinearSVR(C=0.5)
        model_3 = LinearSVR(C=1)
        model_4 = LinearSVR(C=2)
        model_5 = LinearSVR(C=10)
        model_names = ['C=0.1', 'c=0.5', 'c=1', 'c=2', 'c=10']  # 不同模型的名称列表
        model_dic = [model_1, model_2, model_3, model_4, model_5]  # 不同回归模型对象的集合
        cv_score_list = []  # 交叉检验结果列表
        pre_y_list = []  # 各个回归模型预测的y值列表
        for model in model_dic:  # 读出每个回归模型对象
            scores = cross_val_score(model, X, y, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
            cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
            pre_y_list.append(model.fit(X, y).predict(X))  # 将回归训练中得到的预测y存入列表
        # 模型效果指标评估
        n_samples, n_features = X.shape  # 总样本量,总特征数
        model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
        model_metrics_list = []  # 回归评估指标列表
        for i in range(5):  # 循环每个模型索引
            tmp_list = []  # 每个内循环的临时结果列表
            for m in model_metrics_name:  # 循环每个指标对象
                tmp_score = m(y, pre_y_list[i])  # 计算每个回归指标结果
                tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
            model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
        df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
        df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
        print('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量=
        print(70 * '-')  # 打印分隔线
        print('cross validation result:')  # 打印输出标题
        print(df1)  # 打印输出交叉检验的数据框
        print(70 * '-')  # 打印分隔线
        print('regression metrics:')  # 打印输出标题
        print(df2)  # 打印输出回归指标的数据框
        print(70 * '-')  # 打印分隔线
        print('short name \t full name')  # 打印输出缩写和全名标题
        print('ev \t explained_variance')
        print('mae \t mean_absolute_error')
        print('mse \t mean_squared_error')
        print('r2 \t r2')
        print(70 * '-')  # 打印分隔线
        # 模型效果可视化
        plt.figure()  # 创建画布
        plt.plot(np.arange(X.shape[0]), y, color='k', label='true y')  # 画出原始值的曲线
        color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
        linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
        for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
            plt.plot(np.arange(X.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
        plt.title('regression result comparison')  # 标题
        plt.legend(loc='upper right')  # 图例位置
        plt.ylabel('real and predicted value')  # y轴标题
        plt.show()  # 展示图像
    if flag_get_feature_importance:
        model = LinearSVR(C=0.5)
        model.fit(train_data, train_target)
        y_pred = model.predict(test_data)
        print(70 * '-')  # 打印分隔线
        print("结果：")
        basic_model_evaluation(model,train_data,train_target,test_data,test_target,y_pred)
        print(70 * '-')  # 打印分隔线
        # 特征重要度
        features = list(predict_feature)
        importances = model.coef_
        indices = np.argsort(importances)[::-1]
        # 输出各个特征的重要度
        print("各个特征的重要度：")
        for i in indices:
            print("{0} ： {1:.3f}".format(features[i], importances[i]))
        print(70 * '-')
        return model


def fit_RFRegressor(df):
    print('-'*70)
    print("随机森林回归")
    feature_cols = ['城市', '区', '室', '卫', '厅', '面积', '朝向', '所属楼层', '总楼层',
                    '是否有阳台', '信息发布人类型', '是否有床', '是否有衣柜', '是否有沙发', '是否有电视', '是否有冰箱', '是否有洗衣机', '是否有空调',
                    '是否有热水器', '是否有宽带', '是否有燃气', '是否有暖气', '最近学校距离', '周边学校个数', '最近医院距离', '周边医院个数']
    x_data = df[feature_cols]
    x_data = pd.get_dummies(x_data)
    y_data = df['价格']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0)
    # n_estimators：森林中树的数量
    # n_jobs  整数 可选（默认=1） 适合和预测并行运行的作业数，如果为-1，则将作业数设置为核心数
    # forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    forest = RandomForestRegressor()
    forest.fit(x_train, y_train)
    forest_y_predict = forest.predict(x_test)
    basic_model_evaluation(forest,x_train,y_train,x_test,y_test,forest_y_predict)#evaluation

    # 下面对训练好的随机森林，完成重要性评估
    # feature_importances_  可以调取关于特征重要程度
    importances = forest.feature_importances_
    print("重要性：", importances)
    x_columns = x_data.columns
    indices = np.argsort(importances)[::-1]
    for f in range(x_train.shape[1]):
        # 对于最后需要逆序排序，我认为是做了类似决策树回溯的取值，从叶子收敛
        # 到根，根部重要程度高于叶子。
        print("%2d) %-*s %f" % (f + 1, 30, x_columns[indices[f]], importances[indices[f]]))
    return forest

def fit_XGBoostRegressor(df):
    print('-' * 70)
    print("XGBoost回归")
    feature_cols = ['城市', '区', '室', '卫', '厅', '面积', '朝向', '所属楼层', '总楼层',
                    '是否有阳台', '信息发布人类型', '是否有床', '是否有衣柜', '是否有沙发', '是否有电视', '是否有冰箱', '是否有洗衣机', '是否有空调',
                    '是否有热水器', '是否有宽带', '是否有燃气', '是否有暖气', '最近学校距离', '周边学校个数', '最近医院距离', '周边医院个数']
    x_data = df[feature_cols]
    x_data = pd.get_dummies(x_data)
    y_data = df['价格']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0)
    xgboost = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0, subsample=0.8, colsample_bytree=0.9,
                               max_depth=7)
    xgboost.fit(x_train, y_train)
    xgboost_y_predict = xgboost.predict(x_test)
    basic_model_evaluation(xgboost, x_train, y_train, x_test, y_test, xgboost_y_predict)  # evaluation

    # 下面对训练好的随机森林，完成重要性评估
    # feature_importances_  可以调取关于特征重要程度
    importances = xgboost.feature_importances_
    print("重要性：", importances)
    x_columns = x_data.columns
    indices = np.argsort(importances)[::-1]
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, x_columns[indices[f]], importances[indices[f]]))
    return xgboost

def fit_ElasticNet(df):
    print('-' * 70)
    print("ElasticNet Regressor")
    param_grid={'alpha': [1.0, 0.5, 0.2,0.1], 'l1_ratio': [0.5, 0.4, 0.3,0.2]}
    grid_search = GridSearchCV(ElasticNet(max_iter=3000,tol=1e-5), param_grid=param_grid, cv=10)
    X = df.drop(['价格', 'lng', 'lat', '最近医院距离', '周边医院个数', '最近学校距离', '周边学校个数', '面积', '总楼层'], axis=1)
    y = df['价格']
    scaler = MinMaxScaler()
    numerical = df[['最近医院距离', '周边医院个数', '最近学校距离', '周边学校个数', '面积', '总楼层']]
    numerical = pd.DataFrame(scaler.fit_transform(numerical),columns=['最近医院距离', '周边医院个数', '最近学校距离', '周边学校个数', '面积', '总楼层'])
    X = pd.concat([X, numerical], axis=1)
    X = pd.concat([X, pd.get_dummies(X['城市'])], axis=1)
    X.drop(['城市', '北京'], axis=1, inplace=True)
    X = pd.concat([X, pd.get_dummies(X['租房网站名称'])], axis=1)
    X = X.drop(['租房网站名称', '赶集网', '区'], axis=1)
    #print(X.head(), y.head())
    selector=SelectFromModel(estimator=ElasticNet()).fit(X, y)
    #print(X.columns)
    for var,b in zip(X.columns,selector.get_support()):#被选择的变量
        if b==True:
            print(var)
    X_new = selector.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state=42, test_size=0.25)
    grid_search.fit(X_train,y_train)
    print("best params:", grid_search.best_params_, "best score:", grid_search.best_score_)
    reg = grid_search.best_estimator_
    y_pred = reg.predict(X_test)
    basic_model_evaluation(reg, X_train, y_train, X_test, y_test, y_pred)
    print(reg.get_params())
    return reg

def fit_StackingRegressor(df):
    print('-' * 70)
    print("Stacking Regressor")
    estimators=[('rf',RandomForestRegressor()),('en',ElasticNet(alpha=0.1,l1_ratio=0.5))]
    reg=StackingRegressor(estimators=estimators,final_estimator=RidgeCV())
    X=df.drop(['价格','lng','lat','最近医院距离','周边医院个数','最近学校距离','周边学校个数','面积','总楼层'],axis=1)
    y=df['价格']
    scaler=MinMaxScaler()
    numerical=df[['最近医院距离','周边医院个数','最近学校距离','周边学校个数','面积','总楼层']]
    numerical=pd.DataFrame(scaler.fit_transform(numerical))
    X=pd.concat([X,numerical],axis=1)
    X=pd.concat([X,pd.get_dummies(X['城市'])],axis=1)
    X.drop(['城市','北京'],axis=1,inplace=True)
    X=pd.concat([X,pd.get_dummies(X['租房网站名称'])],axis=1)
    X=X.drop(['租房网站名称','赶集网','区'],axis=1)
    #print(X.head(),y.head())
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.25)
    reg=reg.fit(X_train,y_train)
    y_pred=reg.predict(X_test)
    basic_model_evaluation(reg,X_train,y_train,X_test,y_test,y_pred)
    return reg



def fit_BaggingRegressor(df):
    print('-'*70)
    print("Bagging Regressor")
    reg = BaggingRegressor(base_estimator=DecisionTreeRegressor(),bootstrap_features=True,n_estimators=50)
    X = df.drop(['价格', 'lng', 'lat', '最近医院距离', '周边医院个数', '最近学校距离', '周边学校个数', '面积', '总楼层'], axis=1)
    y = df['价格']
    scaler = MinMaxScaler()
    numerical = df[['最近医院距离', '周边医院个数', '最近学校距离', '周边学校个数', '面积', '总楼层']]
    numerical = pd.DataFrame(scaler.fit_transform(numerical))
    X = pd.concat([X, numerical], axis=1)
    X = pd.concat([X, pd.get_dummies(X['城市'])], axis=1)
    X.drop(['城市', '北京'], axis=1, inplace=True)
    X = pd.concat([X, pd.get_dummies(X['租房网站名称'])], axis=1)
    X = X.drop(['租房网站名称', '赶集网', '区'], axis=1)
    #print(X.head(), y.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
    reg = reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    basic_model_evaluation(reg, X_train, y_train, X_test, y_test, y_pred)
    return reg

def fit_VotingRegressor(df):
    print('-' * 70)
    print("Voting Regressor")
    estimators = [('rf', RandomForestRegressor()), ('en', ElasticNet(alpha=0.1,l1_ratio=0.5))]
    reg = VotingRegressor(estimators=estimators, weights=[0.7,0.3])
    X = df.drop(['价格', 'lng', 'lat', '最近医院距离', '周边医院个数', '最近学校距离', '周边学校个数', '面积', '总楼层'], axis=1)
    y = df['价格']
    scaler = MinMaxScaler()
    numerical = df[['最近医院距离', '周边医院个数', '最近学校距离', '周边学校个数', '面积', '总楼层']]
    numerical = pd.DataFrame(scaler.fit_transform(numerical))
    X = pd.concat([X, numerical], axis=1)
    X = pd.concat([X, pd.get_dummies(X['城市'])], axis=1)
    X.drop(['城市', '北京'], axis=1, inplace=True)
    X = pd.concat([X, pd.get_dummies(X['租房网站名称'])], axis=1)
    X = X.drop(['租房网站名称', '赶集网', '区'], axis=1)
    #print(X.head(), y.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
    reg = reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    basic_model_evaluation(reg, X_train, y_train, X_test, y_test, y_pred)
    return reg

def basic_model_evaluation(model,train_data,train_target,test_data,test_target,y_pred):
    # explained_variance_score:解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量
    # 的方差变化，值越小则说明效果越差。
    # mean_absolute_error:平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度
    # ，其其值越小说明拟合效果越好。
    # mean_squared_error:均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的
    # 平方和的均值，其值越小说明拟合效果越好。
    # r2_score:判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因
    # 变量的方差变化，值越小则说明效果越差。
    print("train_R2：", model.score(train_data, train_target))
    print("test_R2：", model.score(test_data, test_target))
    print("test_MSE：", mean_squared_error(test_target, y_pred.reshape(-1, 1)))
    print("test_RMSE", math.sqrt(mean_squared_error(test_target, y_pred.reshape(-1, 1))))
    print("explained variance",explained_variance_score(test_target,y_pred.reshape(-1,1)))


## 定义了一个统计函数，方便后续信息统计
def Sta_inf(data):
    print('_min',np.min(data))
    print('_max:',np.max(data))
    print('_mean',np.mean(data))
    print('_ptp',np.ptp(data))
    print('_std',np.std(data))
    print('_var',np.var(data))

if __name__=="__main__":
    house_dataset=read_data()
    EDA(house_dataset)
    #SVMRegressor=fit_SVMRegressor(house_dataset,flag_kernel=False,flag_search_param=False,flag_model_cross_valid=False,flag_get_feature_importance=True)

    #RFRegressor=fit_RFRegressor(house_dataset)
    #XGBoostRegressor=fit_XGBoostRegressor(house_dataset)

    #ElasticRegressor=fit_ElasticNet(house_dataset)
    #StackingRegressor=fit_StackingRegressor(house_dataset)
    #BaggingRegressor=fit_BaggingRegressor(house_dataset)
    #VotingRegressor=fit_VotingRegressor(house_dataset)

