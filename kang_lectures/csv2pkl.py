import pandas as pd
import os

def csv_to_pickle(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 提取文件名（不包括扩展名）
    file_name, _ = os.path.splitext(csv_file)
    
    # 保存为同名的Pickle文件
    pickle_file = f'{file_name}.pkl'
    df.to_pickle(pickle_file)
    
    print(f'CSV文件 {csv_file} 已成功转换为Pickle文件 {pickle_file}')

# 要转换的CSV文件名
csv_file_name = '510050.SH_15.csv'

# 调用函数进行转换
csv_to_pickle(csv_file_name)
