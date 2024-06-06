# 加载必要的库
import numpy as np
import pickle
from pprint import pprint

# 定义加载数据的函数
def load_deap_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

# 加载本地数据
participant_id = 1
data_file = f'data_preprocessed_python/s{participant_id:02d}.dat'
subject = load_deap_data(data_file)
# pprint(subject)
# input("===")
# 提取数据和标签
X = subject['data']
y_valence = subject['labels'][:, 0]  # 效价标签

# 输出原始数据的维度
print(f'原始数据维度: {X.shape}')

# 展示部分数据
# print(f'数据示例: {X[0]}')  # 这里展示第一个样本的数据


