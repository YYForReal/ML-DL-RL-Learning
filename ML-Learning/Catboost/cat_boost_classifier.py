#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   cat_boost_classifier_test.py
@Time    :   2024/09/30 15:11:15
@Author  :   YYForReal 
@Email   :   2572082773@qq.com
@description   :   catboost 分类器训练代码
'''

# 1. 安装必要的库
# pip install pandas sklearn openpyxl matplotlib catboost argparse joblib

# 2. 导入必要的库
import pandas as pd
import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib
from catboost import CatBoostClassifier
import numpy as np

# 设置字体为支持中文的字体，防止中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# 处理命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="CatBoost 分类模型训练")
    
    # 添加 is_all 和 is_train 的默认值为 True，load_model 默认为 None
    parser.add_argument('--is_all', action='store_true', default=True, help='使用全部数据（默认：使用全部数据）')
    parser.add_argument('--is_train', action='store_true', default=True, help='是否训练模型（默认：进行训练）')
    parser.add_argument('--load_model', type=str, default=None, help='加载已训练的模型路径（默认：不加载模型）')
    
    return parser.parse_args()

# 处理极端值（可选）
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def main():
    args = parse_args()

    # 选择文件
    file_name = "2023年总数据" if args.is_all else "2023年数据"
    file_path = f'data/{file_name}.xlsx'

    print(f"读取数据: {file_name}")
    df = pd.read_excel(file_path)
    
    print(f"过滤前的训练集大小: {df.shape}")

    # 过滤掉基本保额和金额为空的记录
    df = df.dropna(subset=['基本保额', '金额', '险种类别', '是否调查'])

    # 新增：过滤掉基本保额和金额小于0的记录，以及险种类别不为‘健康险’或‘意外险’的记录
    df = df[(df['基本保额'] >= 0) & (df['金额'] >= 0) & df['险种类别'].isin(['健康险', '意外险'])]
    
    # 新增：过滤掉“长短险标识”的列，保留值为'S'的记录
    df = df[df['长短险标识'] == 'S']
    
    print(f"过滤后的训练集大小: {df.shape}")

    # 数据预处理
    features = [
        '险种代码', '出险一级原因', '出险结果',
        '医疗责任类别', '基本保额', '医保身份', '金额'
    ]

    # 保证 '基本保额' 是数值型，其他类别型特征转为字符串
    categorical_features = ['险种代码', '出险一级原因', '出险结果', '医疗责任类别', '医保身份']
    for feature in categorical_features:
        df[feature] = df[feature].astype(str)

    # 特征和目标变量 (是否调查，1 或 0)
    X = df[features]
    y = df['是否调查'].astype(int)  # 使用“是否调查”作为目标变量（分类任务）

    # 再次检查是否存在 NaN 数据
    if X.isnull().sum().any() or y.isnull().sum().any():
        raise ValueError("特征或目标变量中存在 NaN 值，请检查数据")

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 检查是否要加载模型
    if args.load_model:
        print(f"加载模型: {args.load_model}")
        cat_classifier = joblib.load(args.load_model)
    else:
        # 训练 CatBoost 分类模型
        if args.is_train:
            categorical_features_indices = [X.columns.get_loc(col) for col in categorical_features]

            print("开始训练 CatBoost 分类模型...")
            cat_classifier = CatBoostClassifier(
                iterations=10000,           # 减少迭代次数，避免过拟合
                learning_rate=0.02,        # 增加学习率
                depth=8,                  # 保持模型深度
                eval_metric='Accuracy',    # 使用准确率作为评估指标
                random_seed=42,
                early_stopping_rounds=1500,  # 提前停止，防止过度拟合
                verbose=100
            )

            cat_classifier.fit(
                X_train, y_train,
                cat_features=categorical_features_indices,
                eval_set=(X_test, y_test),
                early_stopping_rounds=1500
            )

            # 保存模型
            model_path = f'models/catboost_classifier_{file_name}.pkl'
            if not os.path.exists('models'):
                os.makedirs('models')
            joblib.dump(cat_classifier, model_path)
            print(f"模型已保存至: {model_path}")
        else:
            print("未指定模型训练且未加载模型，程序退出。")
            return

    # 预测是否调查
    y_pred_test = cat_classifier.predict(X_test)
    y_pred_train = cat_classifier.predict(X_train)

    # 评估模型在测试集上的表现
    accuracy_test = accuracy_score(y_test, y_pred_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    report_test = classification_report(y_test, y_pred_test)

    print(f"测试集上的评估指标：")
    print(f"准确率 (Accuracy): {accuracy_test}")
    print(f"混淆矩阵 (Confusion Matrix):\n {cm_test}")
    print(f"分类报告 (Classification Report):\n {report_test}")

    # 评估模型在训练集上的表现
    accuracy_train = accuracy_score(y_train, y_pred_train)
    cm_train = confusion_matrix(y_train, y_pred_train)
    report_train = classification_report(y_train, y_pred_train)

    print(f"\n训练集上的评估指标：")
    print(f"准确率 (Accuracy): {accuracy_train}")
    print(f"混淆矩阵 (Confusion Matrix):\n {cm_train}")
    print(f"分类报告 (Classification Report):\n {report_train}")

    # 特征重要性可视化
    feature_importances = cat_classifier.get_feature_importance()
    feature_names = X.columns

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='salmon')
    plt.xlabel('特征重要性')
    plt.ylabel('特征名称')
    plt.title(f'模型特征重要性 (CatBoost Classifier-{file_name})')
    plt.show()

    # 绘制训练集和测试集的分类错误曲线
    evals_result = cat_classifier.get_evals_result()

    train_error = evals_result['learn']['Accuracy']
    test_error = evals_result['validation']['Accuracy']

    plt.figure(figsize=(10, 6))
    plt.plot(train_error, label='训练集 准确率')
    plt.plot(test_error, label='测试集 准确率')
    plt.title(f'训练和测试集的准确率曲线 (CatBoost-{file_name})')
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
