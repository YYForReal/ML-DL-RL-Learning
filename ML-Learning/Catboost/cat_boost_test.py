#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   cat_boost_classifier_test.py
@Time    :   2024/09/30 15:11:15
@Author  :   YYForReal 
@Email   :   2572082773@qq.com
@description   :   catboost 回归测试代码
'''
# 1. 安装必要的库
# pip install pandas sklearn openpyxl matplotlib catboost argparse joblib numpy

# python .\cat_boost_test.py --load_model  ./models/catboost_model_2023年总数据.pkl --load_test_file ./data/test_data.xlsx --output_test_file ./data/test_result.xlsx

# 2. 导入必要的库
import pandas as pd
import argparse
import os
import joblib
import numpy as np  # 导入 numpy
import matplotlib.pyplot as plt
import matplotlib
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# 设置字体为支持中文的字体，防止中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# 处理命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="CatBoost 模型测试")
    
    # 加载模型路径
    parser.add_argument('--load_model', type=str, required=True, help='加载已训练的模型路径')
    
    # 加载测试文件
    parser.add_argument('--load_test_file', type=str, required=True, help='加载测试数据文件路径')
    
    # 输出测试结果文件
    parser.add_argument('--output_test_file', type=str, required=True, help='输出预测结果文件路径')
    
    return parser.parse_args()

# 读取测试数据
def load_test_data(file_path):
    try:
        df = pd.read_excel(file_path)
        print(f"读取测试数据: {file_path}")
        print(f"数据集大小: {df.shape}")
        return df
    except Exception as e:
        print(f"读取测试数据失败: {e}")
        raise

# 处理测试数据：过滤无效数据
def process_test_data(df):
    # 过滤掉空值记录
    df_before = df.shape[0]

    # 过滤掉基本保额和金额为空的记录
    df = df.dropna(subset=['基本保额', '金额', '险种类别'])

    # 新增：过滤掉基本保额和金额小于0的记录，以及险种类别不为‘健康险’或‘意外险’的记录
    df = df[(df['基本保额'] >= 0) & (df['金额'] >= 0) & df['险种类别'].isin(['健康险', '意外险'])]
    
    # 新增：过滤掉“长短险标识”的列，保留值为'S'的记录
    df = df[df['长短险标识'] == 'S']

    df_after = df.shape[0]
    print(f"过滤了 {df_before - df_after} 行数据，剩余 {df_after} 行有效数据")
    
    return df

# 主流程
def main():
    args = parse_args()

    # 1. 加载已训练的模型
    if not os.path.exists(args.load_model):
        raise FileNotFoundError(f"模型文件未找到: {args.load_model}")
    
    print(f"加载模型: {args.load_model}")
    cat_regressor = joblib.load(args.load_model)

    # 2. 加载测试数据
    df_test = load_test_data(args.load_test_file)

    # 3. 处理测试数据（过滤金额、基本保额小于0的记录，以及险种类别过滤）
    df_test = process_test_data(df_test)

    # 定义需要的特征
    features = ['险种代码', '出险一级原因', '出险结果', '医疗责任类别', '基本保额', '医保身份']

    # 如果过滤后没有数据，提示用户
    if df_test.shape[0] == 0:
        raise ValueError("测试数据过滤后为空，请检查数据")

    # 将类别型特征转换为字符串类型
    for feature in features:
        df_test[feature] = df_test[feature].astype(str)

    # 特征变量
    X_test = df_test[features]
    
    # 4. 使用已训练模型进行预测
    print("\n开始预测...")
    try:
        y_pred_cat = cat_regressor.predict(X_test)
        # 截断负值为 0
        y_pred_cat = np.clip(y_pred_cat, 0, None)
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return

    # 将预测结果存储到DataFrame中
    df_test['预测金额'] = y_pred_cat

    # 5. 计算误差指标
    if '金额' in df_test.columns:
        # 平均误差
        mean_error = np.mean(df_test['金额'] - df_test['预测金额'])
        # 中位数误差
        median_error = np.median(df_test['金额'] - df_test['预测金额'])
        # 均方误差
        mse = mean_squared_error(df_test['金额'], df_test['预测金额'])
        
        print(f"平均误差: {mean_error}")
        print(f"中位数误差: {median_error}")
        print(f"均方误差: {mse}")
    else:
        print("测试数据中不包含真实金额列，无法计算误差指标。")

    # 6. 将结果输出到指定的文件中
    output_file = args.output_test_file
    output_dir = os.path.dirname(output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_test.to_excel(output_file, index=False)
    print(f"预测结果已保存至: {output_file}")

    # 6. 可视化预测结果（若有真实值可以对比）
    if '金额' in df_test.columns:
        print("\n生成可视化对比图...")
        plt.figure(figsize=(10, 6))
        plt.plot(df_test['金额'], label='真实金额', marker='o')
        plt.plot(df_test['预测金额'], label='预测金额', marker='x')
        plt.title('真实金额 vs 预测金额 (CatBoost)')
        plt.xlabel('样本')
        plt.ylabel('金额')
        plt.legend()
        plt.show()
    else:
        print("测试数据中不包含真实金额列，无法进行对比可视化。")

if __name__ == '__main__':
    main()
