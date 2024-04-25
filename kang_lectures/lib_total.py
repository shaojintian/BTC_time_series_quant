import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import talib as ta

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb # 集成算法

# 机器学习

import joblib
import pickle


def factor_selection_by_correlation(data, column_list, corr_threshold): # 根据corr_threshold设置阈值筛选因子
    '''
    输入
        data             : train set
        column_list      : column list of original dataframe (e.g., df_cols)
        corr_threshold   : threshold of the absolute value of correlation between factors (e.g., 0.75)
    '''
    X = data[column_list[:-1]] #从2开始是因为0是etime，1是y或者y_class
    y = data['return']
    # 后面自己的程序需要把这里修改一下，尤其是对于x和y

    # 计算协方差矩阵
    X_corr_matrix = X.corr()

    # 因子列表初始化
    factor_list_1 = [i for i in X_corr_matrix.columns]
    factor_list_2 = [i for i in X_corr_matrix.columns]

    for i in range(0, len(factor_list_1), 1):
        fct_1 = factor_list_1[i]
        for j in range(0, i, 1):
            fct_2 = factor_list_1[j]
            corr_value = X_corr_matrix.iloc[i, j]
            if abs(corr_value) > corr_threshold:  # 如果两个因子彼此之间的相关系数（绝对值）超过设定阈值threshold，则进一步比较它们各自与y的相关系数（绝对值）以实现因子筛选
                corr_1 = np.corrcoef(X[fct_1], y)[0, 1]
                corr_2 = np.corrcoef(X[fct_2], y)[0, 1]
                if (abs(corr_1) < abs(corr_2)) and (fct_1 in factor_list_2):
                    factor_list_2.remove(fct_1) # value大于corr_threshold的继续比较他们和y的相关系数，较小的一半刷掉
    
    return factor_list_2


def generate_etime_close_data_divd_time(bgn_date, end_date, index_code, frequency): # 2022-11-25 edition last edited on 2023-01-17
    # 读取数据
    read_file_path = 'D:/9_quant_course/' + index_code + '_' + frequency + '.xlsx'
    kbars = pd.read_excel(read_file_path)
    kbars['tdate'] = pd.to_datetime(kbars['etime']).dt.date # 这一段需要优化
    kbars['etime'] = pd.to_datetime(kbars['etime'])
    kbars['label'] = '-1'
    
    # 根据区间开始和结束日期截取数据
    bgn_date = pd.to_datetime(bgn_date)
    end_date = pd.to_datetime(end_date)
    for i in range(0, len(kbars), 1): # .strftime('%Y-%m-%d %H:%M:%S')
        if (bgn_date <= kbars.loc[i, 'etime']) and (kbars.loc[i, 'etime'] <= end_date):
            kbars.loc[i, 'label'] = '1'

    # 筛选数据并重置索引
    kbars = kbars[kbars['label'] == '1']
    kbars = kbars.reset_index(drop=True)
    etime_close_data = kbars[['etime', 'tdate', 'close']]
    etime_close_data = etime_close_data.reset_index(drop=True)

    return etime_close_data



path = './'

# 一、读取数据并根据talib生成各项基础因子
start_time = time.time()
file_path =path + '510050.SH_15.pkl' # pickle feather
data_15mins_50etf = pd.read_pickle(file_path).reset_index() # 读取源文件
data_15mins_50etf['timestamp'] = pd.to_datetime(data_15mins_50etf['timestamp'])
data_15mins_50etf = data_15mins_50etf.sort_values(by='timestamp', ascending=True)
data_15mins_50etf = data_15mins_50etf.set_index('timestamp')

t = 1
data_15mins_50etf['return'] = data_15mins_50etf['close'].shift(-t)/data_15mins_50etf['close'] - 1
data_15mins_50etf = data_15mins_50etf.replace([np.nan], 0.0)
print('data_15mins_50etf:\n', data_15mins_50etf)

fct_value = pd.DataFrame() # 新建dataframe，命名为fct_value

# 2、载入因子，观察其corr
fct_file = pd.read_csv(path + 'factor_base_0419.csv').set_index('timestamp')
fct_corr = fct_file.corr()
fct_file['return'] = (data_15mins_50etf['close'].shift(-1)/data_15mins_50etf['close'] - 1).values # 注意这里需要加上.values
fct_file = fct_file.replace([np.nan], 0.0)

column_list = list(set(fct_file.columns) - set(['return']))
fct_corr_05 = factor_selection_by_correlation(fct_file, column_list, 0.5) #

print(fct_corr_05)
print(len(fct_corr_05))