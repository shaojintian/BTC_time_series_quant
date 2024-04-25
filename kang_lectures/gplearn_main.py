import random
import numpy as np
import pandas as pd
import time

from gplearn.genetic import SymbolicTransformer # 符号变换
from gplearn.fitness import make_fitness

from sklearn import metrics as me
import warnings
warnings.filterwarnings('ignore')


def preprocess(file_path, frequency, n_days):
    df = pd.read_pickle(file_path)  # File should be in the form of Pickle

    t = 1
    df['etime'] = pd.to_datetime(df['etime'])
    df['ret'] = df['close'].pct_change(periods=t).shift(-t)
    # 注意这里return也需要给他做norm处理以保证数据无偏

    df.dropna(axis=0, how='any', inplace=True)
    df = df.reset_index(drop=True)

    fields = df.columns[1: ]  # e.g. ['etime', 'open', 'high', 'low', 'close', 'volume', 'ret'][1: ] # 特征

    for col in fields:
        df[col] = df[col].values.astype('float')
    
    X_train = df.drop(columns=['etime', 'ret']).to_numpy()
    y_train = df['ret'].values
    
    return np.nan_to_num(X_train), np.nan_to_num(y_train)


start_time = time.time()
X_train, y_train = preprocess(file_path='./510050_15.pkl', frequency='15', n_days=1)
# 不应该是全局数据，应该是70%，或者60%的数据，留一个验证集，测试集；
# 作业1：x_train和x_test的分离 # xtrain,ytrain,截取其中的一部分数据即可

#"""Activate the Customized Functions"""

func_1 = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'sin', 'cos', 'tan']
#factors = ['tanh', 'elu', 'TA_HT_TRENDLINE', 'TA_HT_DCPERIOD', 'TA_HT_DCPHASE', 'TA_SAR', 'TA_BOP', 'TA_AD', 'TA_OBV', 'TA_TRANGE'] # 算子

# 统计物理，信号与系统；# 把talib里面的，全部加进去；

#gplearn.functions.functions.py 是修改过的源码
from gplearn.functions import _function_map

user_func = func_1 + random.sample(list(_function_map.keys())[14:],10)
print(user_func)
#user_func = list(_function_map.keys())[:2]

#sortino
def _my_metric_sortino(y, y_pred, w): # 打分机制--损失函数--作业2
    #return me.normalized_mutual_info_score(y, y_pred) # 互信息
    # 计算超额回报
    required_return = 0.00
    excess_returns = y - y_pred - required_return
    
    # 只考虑下行波动性，即超额回报为负的部分
    downside_returns = excess_returns[excess_returns < 0]
    
    # 计算下行波动性（下行标准差）
    downside_std = np.sqrt(np.mean(downside_returns**2))
    
    # Sortino Ratio
    sortino = np.mean(excess_returns) / (downside_std + 1e-10)  # 添加小常数以避免除以零
    
    return sortino

def _my_metric_sharpe(y, y_pred, w): # 打分机制--损失函数--作业2
     # 计算模型的平均性能（例如，平均准确度）
    performance = np.mean(y - y_pred)
    
    # 计算模型性能的标准差
    std_dev = np.std(y - y_pred)
    
    # 假设无风险利率为0，这里可以修改为实际的无风险利率
    risk_free_rate = 0.05
    
    # 计算 Sharpe 比率
    sharpe_ratio = performance / (std_dev + 1e-10)  # 添加一个小常数以避免除以零
    
    return sharpe_ratio


user_metric = make_fitness(function=_my_metric_sharpe, greater_is_better=True, wrap=False)

# 作业2：自己尝试着构建以sharpe为基准的metric

#1.2 
ST_gplearn = SymbolicTransformer(population_size=500, # 一次生成因子的数量，
                                 hall_of_fame=100, # 
                                 n_components=100, # 最终输出多少个因子？
                                 generations=3, # 非常非常非常重要！！！--进化多少轮次？3也就顶天了
                                 tournament_size=80,
                                 const_range=None, #(-1, 1),  # critical
                                 init_depth=(2, 5), # 第二重要的一个部位，控制我们公式的一个深度
                                #function_set=user_func, # 输入的算子群
                                 function_set=user_func, # 输入的杂交方式,群
                                #  metric=user_metric, # 提升的点
                                 metric=user_metric, # pearson相关系数：思考的深度不够，自己想办法，把sharpe写进来，本节课的终极作业；spearman
                                 parsimony_coefficient=0.01,
                                 p_crossover=0.4, 
                                 p_subtree_mutation=0.01,
                                 p_hoist_mutation=0.01,
                                 p_point_mutation=0.01,
                                 p_point_replace=0.4,
                                 feature_names=['open', 'high', 'low', 'close', 'volume'], # 注意这里必须有feature_names 把重要算子“聚合化”，
                                 n_jobs=-1,
                                 random_state=1)


#1.3 training
ST_gplearn.fit(X_train, y_train)
# 把这些expression，去给他通过正则表达式匹配，给他函数化。

#"""Rank and Save the Generated Alpha Factors"""


best_programs = ST_gplearn._best_programs
best_programs_dict = {}

for bp in best_programs:
    factor_name = 'alpha_' + str(best_programs.index(bp) + 1)
    best_programs_dict[factor_name] = {'fitness': bp.fitness_, 'expression': str(bp), 'depth': bp.depth_, 'length': bp.length_}

best_programs_frame = pd.DataFrame(best_programs_dict).T
best_programs_frame = best_programs_frame.sort_values(by='fitness', axis=0, ascending=False)
best_programs_frame = best_programs_frame.drop_duplicates(subset=['expression'], keep='first')

print(best_programs_frame)
best_programs_frame.to_csv(f'../gp_cross_overed/fct_gp_cross_overed_{time.time()}.csv')

end_time = time.time()
print('Time Cost:-----    ', end_time-start_time, 'S    --------------')

# TA_HT_DCPERIOD(TA_OBV(TA_HT_DCPERIOD(tanh(low)), high))


