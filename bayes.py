# Bayes估计  bzxysjl@163.com
import numpy as np
from collections import defaultdict


def tranning(data:np.ndarray, labels:np.ndarray):
    """
    bayes训练
    :param data: 训练样本（每行为一个样本）
    :param labels: 对应的分类
    :return:
    """
    dimension = data.shape[1]
    data_size = data.shape[0]
    priori = {}         # 先验概率
    conditional = {}    # 条件概率
    # 以下循环统计次数
    for i in range(data_size):
        priori[labels[i]] = priori.get(labels[i],0)+1
        if labels[i] not in conditional:
            conditional[labels[i]] = {}
        for j in range(dimension):
            if j not in conditional[labels[i]]:
                conditional[labels[i]][j] = {}
            k = data[i,j]
            conditional[labels[i]][j][k] = conditional[labels[i]][j].get(k,0)+1
    # 将次数除以总次数得到对应的概率
    y = 1.0    # 对应课本P51的λ值
    for i in conditional:
        for j in range(dimension):
            for k in conditional[i][j]:
                conditional[i][j][k] = (conditional[i][j][k]+y) / (priori[i]+len(conditional[i][j])*y)
    for k in priori:
        priori[k]=(priori[k]+y)/(data_size+len(priori)*y)
    return priori, conditional


def classify_bayes(sample:np.ndarray, priori:dict, conditional:dict):
    """
    bayes分类
    :param sample: 待分类样本
    :param priori: 先验概率
    :param conditional: 条件概率
    :return: (分类,概率)
    """
    max_probability = -1
    max_label = None
    for c in priori:
        probability = priori[c]
        for j in range(len(sample)):
            probability = probability * conditional[c][j][sample[j]]
        if probability>max_probability:
            max_probability = probability
            max_label = c
    return max_label, max_probability


def test_bayes():
    # 测试数据来源于《统计学习方法》P50例4.1
    all =np.array([[1,'s',-1],[1,'m',-1],[1,'m',1],[1,'s',1],[1,'s',-1],
                   [2,'s',-1],[2,'m',-1],[2,'m',1],[2,'l',1],[2,'l',1],
                   [3,'l',1],[3,'m',1],[3,'m',1],[3,'l',1],[3,'l',-1]])
    data = all[:,0:2]
    labels = all[:,-1]
    a,b = tranning(data,labels)
    print('training result is:')
    print(a)
    print(b)
    print('test a sample')
    sample = np.array([2,'s'])
    lable, probability = classify_bayes(sample, a, b)
    print('smaple ',sample ,' is classified as ',lable,' with probability',probability)


if __name__=='__main__':
    test_bayes()

