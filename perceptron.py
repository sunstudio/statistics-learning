# 感知机学习算法
# sunjilei  20180708

import numpy as np


def perceptron(x, y, rate):
    """
    x为样本数，二维数组，其中每列为一个样本
    y为对应的分类，一维数组
    返回三元组（是否成功，w，b）
    """
    # 参数检查（数组维数和长度）
    xshape, yshape = checkShape(x, y)
    # 初始化w和b
    w = np.zeros(xshape[0])
    b = 0
    xt = x.T
    mistake = True      # 是否有误分类样本
    count = 0           # 总迭代次数，超过一定数量时认为样本非线性可分，退出算法
    while mistake:
        if count > 10000:
            return False, w, b
        mistake = False
        # 对样本（x的列）循环，找到一个误分类数据
        for i in range(xshape[1]):
            sample = xt[i]
            if y[i]*(np.dot(sample, w)+b) <= 0:
                mistake = True
                w = w + rate * y[i]*sample
                b = b + rate * y[i]
                count = count + 1
                print("第{0}次迭代：x={1},w={2},b={3}".format(count, sample, w, b))
    return True, w, b


def perceptron_dual(x, y, rate):
    """
    感知机算法对偶形式
    x为样本数，二维数组，其中每列为一个样本
    y为对应的分类，一维数组
    返回三元组（是否成功，alpha，b）
    """
    # 参数检查（数组维数和长度）
    xshape, yshape = checkShape(x, y)
    # 初始化alpha和b
    alpha = np.zeros(xshape[1])
    b = 0
    xt = x.T
    mistake = True      # 是否有误分类样本
    count = 0           # 总迭代次数，超过一定数量时认为样本非线性可分，退出算法
    # 计算gram矩阵
    gram = np.zeros((xshape[1], xshape[1]))
    for i in range(xshape[1]):
        for j in range(xshape[1]):
            gram[i][j ] = np.dot(xt[i], xt[j])
    print('gram is \r\n{0}', gram)
    while mistake:
        if count > 10000:
            return False, alpha, b
        mistake = False
        # 对样本（x的列）循环，找到一个误分类数据
        for i in range(xshape[1]):
            sample = xt[i]
            temp = np.zeros(xshape[0])
            for j in range(xshape[1]):
                temp = temp + alpha[j] * y[j] * xt[j]
            if y[i]*(np.dot(temp, sample)+b) <= 0:
                mistake = True
                alpha[i] = alpha[i] + 1
                b = b + rate * y[i]
                count = count + 1
                print("第{0}次迭代：x={1},alpha={2},b={3}".format(count, sample, alpha, b))
    return True, alpha, b


def checkShape(x, y):
    xshape = x.shape
    yshape = y.shape
    if not(isinstance(x, np.ndarray) or isinstance(y, np.ndarray)):
        raise Exception("参数类型不正确，必须为ndarray")
    if x.ndim != 2 or y.ndim != 1:
        raise Exception("数组维数不正确")
    if xshape[1] != yshape[0]:
        raise Exception("x,y个数不匹配")
    return xshape, yshape


def main():
    n = int(input("请输入样本数量："))
    m = int(input("请输入样本维数："))
    x = np.zeros((n, m))
    y = np.zeros(n)
    for i in range(n):
        sample = input("第{0}个样本值（逗号隔开）：".format(i+1))
        t = np.fromstring(sample, dtype=float, sep=',')
        if t.size != m:
            raise Exception("样本维数不正确")
        x[i] = t
        y[i] = int(input("第{0}个分类（1或-1)：".format(i+1)))
    # result = perceptron(x.T, y, 1.0)
    result = perceptron_dual(x.T, y, 1.0)
    if result[0]:
        print("分类成功。(w或alpah)={0},b={1}".format(result[1],result[2]))
    else:
        print("分类失败。")


if __name__ == "__main__":
    main()




