import os
import pickle
import numpy as np

def load_wesad_data(participant_id, data_path='WESAD'):
    file_path = os.path.join(data_path, f'S{participant_id}', f'S{participant_id}.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

# 加载特定受试者的数据
participant_id = 2  # 示例中加载第2个受试者的数据
data = load_wesad_data(participant_id)

# 查看数据结构
# print(data.keys())  # dict_keys(['signal', 'label', 'subject'])
# print(data['signal'].keys())  # dict_keys(['chest', 'wrist'])

print("data.keys():",data.keys())  # dict_keys(['signal', 'label', 'subject']")
print("data['signal'].keys():",data['signal'].keys())  # dict_keys(['chest', 'wrist'])
print("data['signal']['chest'].keys():",data['signal']['chest'].keys())  # data['signal']['chest'].keys(): dict_keys(['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp'])
print("data['signal']['wrist'].keys():",data['signal']['wrist'].keys())  # data['signal']['wrist'].keys(): dict_keys(['ACC', 'BVP', 'EDA', 'TEMP'])
# data['signal']['chest'] 包含胸部传感器的数据，包括加速度计（ACC）、心电图（ECG）、肌电图（EMG）、皮电反应（EDA）、温度（Temp）和呼吸（Resp）。
# data['signal']['wrist'] 包含手腕传感器的数据，包括加速度计（ACC）、血容量脉搏（BVP）、皮电反应（EDA）和温度（TEMP）。



# 检查每个特征的长度
def check_feature_lengths(data):
    
    chest_data = data['signal']['chest']
    wrist_data = data['signal']['wrist']
    lengths = {'chest_'+key: len(chest_data[key]) for key in chest_data.keys()}
    lengths.update({'wrist_'+key: len(wrist_data[key]) for key in wrist_data.keys()})
    return lengths

feature_lengths = check_feature_lengths(data)
print("各特征的长度: ", feature_lengths)
# 各特征的长度:  {'chest_ACC': 4255300, 'chest_ECG': 4255300, 'chest_EMG': 4255300, 'chest_EDA': 4255300, 'chest_Temp': 4255300, 'chest_Resp': 4255300, 'wrist_ACC': 194528, 'wrist_BVP': 389056, 'wrist_EDA': 24316, 'wrist_TEMP': 24316}