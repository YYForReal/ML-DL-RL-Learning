#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   cat_boost_classifier_test.py
@Time    :   2024/09/30 15:11:15
@Author  :   YYForReal 
@Email   :   2572082773@qq.com
@description   :   catboost 分类器测试代码
'''
# 1. 安装必要的库
# pip install pandas sklearn openpyxl matplotlib catboost argparse joblib numpy

# python .\cat_boost_classifier_test.py --load_model  ./models/catboost_classifier_2023年总数据.pkl --load_test_file ./data/test_result.xlsx --output_test_file ./data/test_result_classify.xlsx

# 2. 导入必要的库
import pandas as pd
import argparse
import os
import joblib
import numpy as np  # 导入 numpy
import matplotlib.pyplot as plt
import matplotlib
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 设置字体为支持中文的字体，防止中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# 处理命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="CatBoost 分类模型测试")
    
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
    df_before = df.shape[0]

    # 检查是否存在 '预测金额' 列
    if '预测金额' not in df.columns:
        raise KeyError("'预测金额' 列在数据中不存在，请检查输入文件。")
    
    # 将 '金额' 列重命名为 '真实金额'
    df = df.rename(columns={'金额': '真实金额'})
    
    # 将 '预测金额' 列重命名为 '金额' 作为当前处理的金额
    df['金额'] = df['预测金额']
    
    # 过滤掉基本保额和预测金额为空的记录
    df = df.dropna(subset=['基本保额', '金额', '险种类别'])

    # 新增：过滤掉基本保额和预测金额小于0的记录，以及险种类别不为‘健康险’或‘意外险’的记录
    df = df[(df['基本保额'] >= 0) & (df['金额'] >= 0) & df['险种类别'].isin(['健康险', '意外险'])]
    
    # 新增：过滤掉“长短险标识”的列，保留值为'S'的记录
    df = df[df['长短险标识'] == 'S']

    df_after = df.shape[0]
    print(f"过滤了 {df_before - df_after} 行数据，剩余 {df_after} 行有效数据")
    
    return df

# 主流程
def main():
    args = parse_args()

    # 1. 加载已训练的分类模型
    if not os.path.exists(args.load_model):
        raise FileNotFoundError(f"模型文件未找到: {args.load_model}")
    
    print(f"加载模型: {args.load_model}")
    cat_classifier = joblib.load(args.load_model)

    # 2. 加载测试数据
    df_test = load_test_data(args.load_test_file)

    # 3. 处理测试数据（过滤金额、基本保额小于0的记录，以及险种类别过滤）
    df_test = process_test_data(df_test)

    # 定义需要的特征
    features = ['险种代码', '出险一级原因', '出险结果', '医疗责任类别', '基本保额', '医保身份', '金额']

    # 如果过滤后没有数据，提示用户
    if df_test.shape[0] == 0:
        raise ValueError("测试数据过滤后为空，请检查数据")

    # 将类别型特征转换为字符串类型
    for feature in ['险种代码', '出险一级原因', '出险结果', '医疗责任类别', '医保身份']:
        df_test[feature] = df_test[feature].astype(str)

    # 特征变量
    X_test = df_test[features]
    
    # 4. 使用已训练分类模型进行是否调查的预测
    print("\n开始预测是否调查...")
    try:
        y_pred_cat = cat_classifier.predict(X_test)
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return

    # 将预测结果存储到 DataFrame 中
    df_test['预测是否调查'] = y_pred_cat

    # 5. 将 '金额' 列重命名为 '预测金额'，并保留 '真实金额'
    df_test = df_test.rename(columns={'金额': '预测金额'})

    # 6. 输出分类结果到指定的文件中
    output_file = args.output_test_file
    output_dir = os.path.dirname(output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_test.to_excel(output_file, index=False)
    print(f"预测结果已保存至: {output_file}")

    # 7. 分类评估（如有实际的‘是否调查’标签可以对比）
    if '是否调查' in df_test.columns:
        accuracy = accuracy_score(df_test['是否调查'], df_test['预测是否调查'])
        cm = confusion_matrix(df_test['是否调查'], df_test['预测是否调查'])
        report = classification_report(df_test['是否调查'], df_test['预测是否调查'])

        print(f"分类准确率: {accuracy}")
        print(f"混淆矩阵:\n{cm}")
        print(f"分类报告:\n{report}")
    else:
        print("测试数据中不包含真实的‘是否调查’列，无法进行分类评估。")

    # 8. 可视化混淆矩阵（若有实际的‘是否调查’标签可以对比）
    if '是否调查' in df_test.columns:
        plt.matshow(cm, cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        plt.ylabel('真实值')
        plt.xlabel('预测值')
        plt.show()
    else:
        print("测试数据中不包含真实的‘是否调查’列，无法进行混淆矩阵的可视化。")

if __name__ == '__main__':
    main()
