#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   cat_boost_classifier_test.py
@Time    :   2024/09/30 15:11:15
@Author  :   YYForReal 
@Email   :   2572082773@qq.com
@description   :   针对输出测试集绘制特定标签类型的散点图，并应用KMeans聚类取中心绘制核心分隔线。
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.font_manager as fm

# 设置字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取并筛选数据
def load_filtered_data(file_path, target_codes):
    df = pd.read_excel(file_path)
    # 根据 '险种代码' 列进行筛选
    df_filtered = df[df['险种代码'].isin(target_codes)]
    print(f"筛选后的数据大小: {df_filtered.shape}")
    return df_filtered

# 绘制散点图并绘制总数据集的分割线
def plot_scatter_with_lines(df, target_codes, total_lines=None):
    plt.figure(figsize=(10, 6))

    # 获取调色板
    colors = plt.get_cmap('tab10')

    # 绘制散点图
    for i, code in enumerate(target_codes):
        code_data = df[df['险种代码'] == code]
        plt.scatter(code_data.index, code_data['预测金额'], color=colors(i % 10), label=f'险种代码: {code}', alpha=0.6)

    # 绘制总数据集的分割线
    if total_lines:
        for line in total_lines:
            plt.axhline(y=line, color='r', linestyle='--', label=f'总分割线: {line:.2f}')

    plt.xlabel('样本')
    plt.ylabel('预测金额')
    plt.title('不同险种代码的预测金额分布及总分割线')
    plt.legend()
    plt.show()

# 使用 KMeans 找到最优的三条分割线
def find_optimal_lines(df):
    # 保证样本数量大于或等于 4
    if len(df) < 4:
        print(f"样本数量过少，无法找到 3 条分割线。当前样本数量: {len(df)}")
        return None

    # 提取 '预测金额' 列
    y_pred = df['预测金额'].values.reshape(-1, 1)

    # 使用 KMeans 聚类，聚为 4 类
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(y_pred)
    
    # 获取聚类中心
    centers = sorted(kmeans.cluster_centers_.flatten())
    
    # 返回三条分割线
    return centers[1], centers[2], centers[3]

# 写入分割线到小 Excel 文件
def save_lines_to_small_excel(lines_data, file_path):
    # 将数据存储为 DataFrame
    df_lines = pd.DataFrame(lines_data, columns=['险种代码', '分割线1', '分割线2', '分割线3'])
    # 保存到Excel
    df_lines.to_excel(file_path, index=False)
    print(f"分割线已保存至: {file_path}")

# 主流程
def main():
    # 险种代码列表
    target_codes = ['967', '968', '970', '975', 'D7X', '971', '966', '965', 'D7I', '845', 
                    'D6F', '978', '642', '958', '889', 'D3W', '632', 'D5O', '645', '818', 
                    '890', '984']

    # 读取数据文件
    # file_path = 'result/predictions.xlsx'
    file_path = './data/test_result.xlsx'
    
    df = load_filtered_data(file_path, target_codes)

    # 存储每个险种代码及其三条分割线
    lines_data = []

    # 计算每个险种代码的三条分割线
    for code in target_codes:
        code_data = df[df['险种代码'] == code]
        lines = find_optimal_lines(code_data)
        if lines:
            lines_data.append([code, *lines])
            print(f"险种代码 {code} 的三条最优分割线: {lines}")

    # 保存三条分割线到新的 Excel 文件
    small_excel_file = 'result/lines_by_code.xlsx'
    save_lines_to_small_excel(lines_data, small_excel_file)

    # 计算整个数据集的最优三条分割线
    total_lines = find_optimal_lines(df)
    if total_lines:
        print(f"总数据集的三条最优分割线: {total_lines}")

    # 绘制带有分割线的散点图
    plot_scatter_with_lines(df, target_codes, total_lines)

if __name__ == '__main__':
    main()
