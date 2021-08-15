# -*- coding: utf-8 -*-
# @Time    : 2021/8/11 11:06
# @Author  : LIU YI

import re
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
from pylab import *

def model_main(file):
    list_point = []
    list_mean = []
    # search the line including accuracy
    for (num, line) in enumerate(file):
        # 基于GDAS框架的TAKD提取
        m1 = re.search('/*/*/* VALID ', line)
        if m1:
            #固定的TA的acc
            n = re.search(r'(?<=accuracy@1 = )\d+\.?\d*', line)
            if n is not None:
                list_point.append(n.group())# 提取精度数字
    list_point = [float(i[0:-1]) / 100 for i in list_point]
    for i in range(0, len(list_point), 10):
        mean = sum(list_point[i: i + 10]) / 10
        list_mean.append(mean)
    return list_mean


def count_conv_num(input):
    output = []
    for i in input:
        count = 0
        for j in i:
            for k in j:
                if 'sepc' in k[0]:
                    count += 1
        output.append(count)

    return output


if __name__ == '__main__':

    f = './seed-18699-T-08-Aug-at-16-29-47.log'
    f = open(f)
    geno_list = []
    normal_cell = []
    reduction_cell = []
    acc_point = []
    for (num, line) in enumerate(f):

        try:
            x = ast.literal_eval(re.search('({.+})', line).group(0))
            geno_list.append(x)
            normal_cell.append(x['normal'])
            reduction_cell.append(x['reduce'])
        except:
            x = 'no dict'

        # m1 = re.search('/*/*/* evaluate ', line)
        n = re.findall(r"accuracy@1=(.+?)%", line)

        if len(n) is not 0:
            acc_point.append(float(n[0]))  # 提取精度数字


    normal = count_conv_num(normal_cell)
    reduce = count_conv_num(reduction_cell)
    geno_size_list = {}
    total = list(np.sum([normal, reduce], axis=0))

    t = linspace(0, 160, 320)
    L1 = plot(t, total, 'lightsteelblue', label = r'$normal cell$')
    legend()
    # plt.show()

    # L2 = plot(t, reduce, 'cornflowerblue', label = r'$reduce cell$')
    # legend()
    # plt.show()

    xlabel(r'$epochs$')
    ylabel(r'# of conv layers')
    twinx()

    L3 = plot(t, acc_point, 'rosybrown', label=r'$accuracy$')
    ylabel(r'accuracy')
    # legend(handles=[L1, L3], loc=9, )
    legend()
    show()

    for one in total:
        geno_size_list[one] = geno_list[total.index(one)]

    print(geno_size_list)

    # print(geno_list)

    #
    # filePath = r"E:\AutoDL-Projects\exps/algos\log/new_log/plane2/cifar10"  # 文件夹路径
    # fileList = os.listdir(filePath)
    # linelist = []
    # filename = []
    # for file in fileList:
    #     f = open(os.path.join(filePath, file))
    #     linelist.append(model_main(f))
    #     n = file.split('.')
    #     filename.append(n[0])
    #     f.close()
    # cmap = plt.get_cmap('viridis')
    # colors = cmap(np.linspace(0, 1, len(linelist))) # 颜色变化
    # markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X',
    #            'D', 'd', '|', '_']
    # for index in range(len(linelist)):
    #     plt.plot(linelist[index], color=colors[index], marker=markers[index])
    # plt.xlabel('Epoch')
    # plt.ylabel('accuracy')
    # plt.title('Accuracy')
    # plt.legend(filename, loc='lower right')
    # plt.show()
