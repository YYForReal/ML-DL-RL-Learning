#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   cat_boost_classifier_test.py
@Time    :   2024/09/30 15:11:15
@Author  :   YYForReal 
@Email   :   2572082773@qq.com
@description   :   catboost 回归训练代码
'''

# 1. 安装必要的库
# pip install pandas sklearn openpyxl matplotlib catboost argparse joblib

# 2. 导入必要的库
import pandas as pd
import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import matplotlib
from catboost import CatBoostRegressor
import numpy as np

# 设置字体为支持中文的字体，防止中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# 处理命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="CatBoost 模型训练")

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
    df = df.dropna(subset=['基本保额', '金额', '险种类别'])

    # 新增：过滤掉基本保额和金额小于0的记录，以及险种类别不为‘健康险’或‘意外险’的记录
    df = df[(df['基本保额'] >= 0) & (df['金额'] >= 0) & df['险种类别'].isin(['健康险', '意外险'])]

    # 新增：过滤掉“长短险标识”的列，保留值为'S'的记录
    df = df[df['长短险标识'] == 'S']

    print(f"过滤后的训练集大小: {df.shape}")
    input("====")
    # 数据预处理
    features = [
        '险种代码', '出险一级原因', '出险结果',
        '医疗责任类别', '基本保额', '医保身份'
    ]

    # 保证 '基本保额' 是数值型，其他类别型特征转为字符串
    categorical_features = ['险种代码', '出险一级原因', '出险结果', '医疗责任类别', '医保身份']
    for feature in categorical_features:
        df[feature] = df[feature].astype(str)

    # 确保目标变量为非负数值
    df['金额'] = df['金额'].apply(lambda x: max(x, 0))  # 将负数替换为0，确保金额为非负值

    # 特征和目标变量
    X = df[features]
    y = df['金额']

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
        cat_regressor = joblib.load(args.load_model)
    else:
        # 训练 CatBoost 回归模型
        if args.is_train:
            categorical_features_indices = [X.columns.get_loc(col) for col in categorical_features]

            print("开始训练 CatBoost 模型...")
            cat_regressor = CatBoostRegressor(
                iterations=10000,           # 减少迭代次数，避免过拟合
                learning_rate=0.02,        # 增加学习率
                depth=8,                  # 保持模型深度
                eval_metric='RMSE',
                random_seed=42,
                early_stopping_rounds=1500,  # 提前停止，防止过度拟合
                verbose=100
            )

            cat_regressor.fit(
                X_train, y_train,
                cat_features=categorical_features_indices,
                eval_set=(X_test, y_test),
                early_stopping_rounds=1500
            )

            # 保存模型
            model_path = f'models/catboost_model_{file_name}.pkl'
            if not os.path.exists('models'):
                os.makedirs('models')
            joblib.dump(cat_regressor, model_path)
            print(f"模型已保存至: {model_path}")
        else:
            print("未指定模型训练且未加载模型，程序退出。")
            return

    # 预测金额
    y_pred_test = cat_regressor.predict(X_test)
    y_pred_train = cat_regressor.predict(X_train)

    # 评估模型在测试集上的表现
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    medae_test = median_absolute_error(y_test, y_pred_test)

    print(f"测试集上的评估指标：")
    print(f"均方误差 (MSE): {mse_test}")
    print(f"平均绝对误差 (MAE): {mae_test}")
    print(f"中位数绝对误差 (MedAE): {medae_test}")

    # 评估模型在训练集上的表现
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    medae_train = median_absolute_error(y_train, y_pred_train)

    print(f"\n训练集上的评估指标：")
    print(f"均方误差 (MSE): {mse_train}")
    print(f"平均绝对误差 (MAE): {mae_train}")
    print(f"中位数绝对误差 (MedAE): {medae_train}")

    # 确保输出不为负值
    y_pred_test = np.clip(y_pred_test, 0, None)  # 将负值剪裁为 0

    # 特征重要性可视化
    feature_importances = cat_regressor.get_feature_importance()
    feature_names = X.columns

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='salmon')
    plt.xlabel('特征重要性')
    plt.ylabel('特征名称')
    plt.title(f'模型特征重要性 (CatBoost-{file_name})')
    plt.show()

    # 可视化真实值和预测值（测试集）
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values[:100], label='真实金额', marker='o')
    plt.plot(y_pred_test[:100], label='预测金额', marker='x')
    plt.title(f'真实金额 vs 预测金额 (测试集 - CatBoost-{file_name})')
    plt.xlabel('样本')
    plt.ylabel('金额')
    plt.legend()
    plt.show()

    # 绘制训练集和测试集的误差曲线图
    evals_result = cat_regressor.get_evals_result()

    # 提取训练集和测试集的 RMSE
    train_error = evals_result['learn']['RMSE']
    test_error = evals_result['validation']['RMSE']

    plt.figure(figsize=(10, 6))
    plt.plot(train_error, label='训练集 RMSE')
    plt.plot(test_error, label='测试集 RMSE')
    plt.title(f'训练和测试集的 RMSE 曲线 (CatBoost-{file_name})')
    plt.xlabel('迭代次数')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

    # 测试模型预测
    test_data_1 = {
        '险种代码': '692',
        '出险一级原因': '2-疾病',
        '出险结果': '医疗+津贴',
        '医疗责任类别': '住院',
        '基本保额': 60000,
        '医保身份': '有医保'
    }

    test_data_2 = {
        '险种代码': '965',
        '出险一级原因': '2-疾病',
        '出险结果': '医疗+津贴',
        '医疗责任类别': '住院',
        '基本保额': 30000,
        '医保身份': '混合医保'
    }

    print(f"测试数据1的预测金额: {test_model(cat_regressor, test_data_1, features)}")
    print(f"测试数据2的预测金额: {test_model(cat_regressor, test_data_2, features)}")

# 定义测试函数
def test_model(model, input_data, features):
    input_df = pd.DataFrame([input_data], columns=features)
    for feature in features:
        input_df[feature] = input_df[feature].astype(str)
    predicted_amount = model.predict(input_df)
    return predicted_amount[0]  # 预测原始金额

if __name__ == '__main__':
    main()
