#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 22:15
# @Author  : JJkinging
# @File    : test.py
from matplotlib import pyplot as plt
import numpy as np

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# x1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]  # 8：2 32 8
# y1 = ['1/8', '2/8', '2/8', '2/8', '3/8', '3/8', '3/8', '3/8', '4/8', '4/8', '4/8', '5/8',
#       '6/8', '7/8', '7/8', '7/8']
# plt.plot(x1, y1, label='the numbers of acc')
# plt.xlabel('第一类问题训练数量')
# plt.ylabel('预测正确数量/总数量')
# plt.legend(loc='best')
# plt.show()


# x1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]  # 8：2 32 9
# y1 = ['0/9', '1/9', '1/9', '2/9', '3/9', '3/9', '4/9', '4/9', '4/9', '5/9', '5/9', '6/9',
#       '7/9', '7/9', '8/9', '8/9']
#
# plt.plot(x1, y1, label='the numbers of acc')
# plt.xlabel('第二类问题训练数量')
# plt.ylabel('预测正确数量/总数量')
# plt.legend(loc='best')
# plt.show()

# x1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33]  # 8：2 33 9
# y1 = ['1/9', '2/9', '2/9', '2/9', '3/9', '3/9', '4/9', '4/9', '4/9', '5/9', '5/9', '6/9',
#       '7/9', '7/9', '7/9', '8/9']
#
# plt.plot(x1, y1, label='the numbers of acc')
# plt.xlabel('第三类问题训练数量')
# plt.ylabel('预测正确数量/总数量')
# plt.legend(loc='best')
# plt.show()

# x1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33]  # 8：2 33 9
# y1 = ['2/9', '2/9', '3/9', '3/9', '3/9', '3/9', '4/9', '4/9', '4/9', '5/9', '5/9', '6/9',
#       '7/9', '7/9', '7/9', '7/9']
#
# plt.plot(x1, y1, label='the numbers of acc')
# plt.xlabel('第四类问题训练数量')
# plt.ylabel('预测正确数量/总数量')
# plt.legend(loc='best')
# plt.show()

# x1 = [2, 4, 6, 8, 10, 12, 14, 16, 17]  # 8：2 17 5
# y1 = ['0/5', '1/5', '1/5', '2/5', '2/5', '2/5', '3/5', '3/5', '3/5']
#
# plt.plot(x1, y1, label='the numbers of acc')
# plt.xlabel('第五类问题训练数量')
# plt.ylabel('预测正确数量/总数量')
# plt.legend(loc='best')
# plt.show()

# x1 = [2, 4, 6, 8, 10, 12, 13]  # 8：2 13 4
# y1 = ['0/4', '1/4', '1/4', '2/4', '2/4', '2/4', '2/4']
#
# plt.plot(x1, y1, label='the numbers of acc')
# plt.xlabel('第六类问题训练数量')
# plt.ylabel('预测正确数量/总数量')
# plt.legend(loc='best')
# plt.show()

# x1 = [2, 4, 6, 8, 10, 12]  # 8：2 12 4
# y1 = ['0/4', '1/4', '1/4', '2/4', '2/4', '2/4']
#
# plt.plot(x1, y1, label='the numbers of acc')
# plt.xlabel('第八类问题训练数量')
# plt.ylabel('预测正确数量/总数量')
# plt.legend(loc='best')
# plt.show()

# x1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 31]  # 8：2 31 8
# y1 = ['2/8', '2/8', '2/8', '2/8', '3/8', '3/8', '4/8', '4/8', '4/8', '4/8', '4/8', '5/8',
#       '6/8', '7/8', '7/8']
#
# plt.plot(x1, y1, label='the numbers of acc')
# plt.xlabel('第九类问题训练数量')
# plt.ylabel('预测正确数量/总数量')
# plt.legend(loc='best')
# plt.show()


# x1 = [10, 20, 30]  # 8：2 228 58
x1 = np.linspace(10, 220, num=22, dtype=np.int)
y1 = [round(2 / 58, 4), round(6 / 58, 4), round(10 / 58, 4), round(16 / 58, 4), round(20 / 58, 4), round(24 / 58, 4),
      round(28 / 58, 4), round(32 / 58, 4), round(33 / 58, 4), round(35 / 58, 4), round(38 / 58, 4), round(40 / 58, 4),
      round(41 / 58, 4),
      round(43 / 58, 4), round(47 / 58, 4), round(49.5 / 58, 4), round(50 / 58, 4), round(52.5 / 58, 4), round(53 / 58, 4),
      round(53.1 / 58, 4),
      round(53.5 / 58, 4), round(54 / 58, 4)]

plt.plot(x1, y1, label='Accuracy', color='red')
plt.xlabel('训练问题数量')
plt.ylabel('准确率')
plt.legend(loc='best')
plt.show()
