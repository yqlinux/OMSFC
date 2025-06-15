# -*- coding:utf-8 -*-
"""
作者：DHX
日期：2021年12月25日
"""
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt
from origin3_codes import graph_ranks

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['lines.linewidth'] = 10

def compute_cd(avranks, n, alpha=0.05):
    """
    计算临界差异值
    :param avranks: 平均排名列表
    :param n: 数据集数量
    :param alpha: 显著性水平
    :return: 临界差异值
    """
    k = len(avranks)
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, float('inf'))
    print(q_alpha)
    return q_alpha * np.sqrt(k * (k + 1) / (6 * n))

names = ['BOOMC','RAOMS','OGS','OSS','OSC','3WD_HD','MSCRS']

# avranks =  [1.75000,2.31250,5.28125,4.87500,4.53125,5.12500,4.12500] # RFAcc
# avranks =  [1.90625,2.09375,5.28125,4.78125,4.62500,4.53125,4.78125] # DTAcc
# avranks =  [1.71875,2.65625,5.00000,5.46875,5.00000,4.65625,3.50000] # SVMAcc
# avranks =  [1.65625,2.21875,5.21875,5.00000,4.84375,4.53125,4.53125] # GBMAcc
avranks =  [1.96875,2.71875,4.31250,4.68750,4.59375,5.03125,4.68750] # XGBAcc

cd = compute_cd(avranks, 16)
print(f"Critical Difference: {cd}")

# 创建图形
plt.figure(figsize=(15, 8))
graph_ranks(avranks, names=names, cd=cd, cdmethod=None, width=15, reverse=True, lowv=1, highv=6, textspace=1.5)
plt.show()


