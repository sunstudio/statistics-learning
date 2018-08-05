import numpy as np
import operator
import matplotlib.pyplot as plt
import os

# 以下代码为machine learning in action书第2章的例子


def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels


def knn(sample, dataSet, labels, k):
    """
    kNN（k近邻）算法，根据最近的k个点的大多数类别来确定要识别样本的类别。
    :param sample: 要识别的样本
    :param dataSet: 训练样本
    :param labels: 训练样本对应的分类
    :param k: k值，即取多少个近邻
    :return: 要识别的样本所归属的分类
    """
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(sample, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    sortedDistance = distance.argsort()
    classCount={}
    for i in range(k):
        label = labels[sortedDistance[i]]
        classCount[label] = classCount.get(label,0)+1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def simpleKnnTest():
    group, labels = createDataSet()
    r = knn([0, 0], group, labels, 3)
    print('class is ', r)


def datingFile2Matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    lineCount = len(lines)
    dataSet = np.zeros((lineCount, 3))
    labels = []
    index = 0
    for line in lines:
        line = line.strip()
        words = line.split('\t')
        dataSet[index, :] = words[0:3]
        labels.append(int(words[-1]))
        index += 1
    return dataSet, labels


def datingDataFigure():
    a, b = datingFile2Matrix('data/datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(a[:, 0], a[:, 1], 30, np.array(b), '.')
    plt.show()


def autoNorm(dataset):
    """
    归一化，将样本值归一化成0到1之间的值
    :param dataset:样本数据
    :return:归一化后的样本数据
    """
    minValues = dataset.min(0)
    maxValues = dataset.max(0)
    rangeValues = maxValues - minValues
    n = dataset.shape[0]
    diffMat = dataset - np.tile(minValues, (n, 1))
    diffMat = diffMat / np.tile(rangeValues, (n, 1))
    return diffMat,  rangeValues, minValues


def testFile2Matrix():
    a, b = datingFile2Matrix('data/datingTestSet2.txt')
    print(a[0:10])
    print(b[0:10])


def testAutoNorm():
    a = np.array([[1.2, 1.6, 201.0, 2234.7],
                  [1.1, 0.9, 322.5, 2311.2],
                  [0.9, 1.0, 257.3, 1990.4]])
    norm0, range0, min0 = autoNorm(a)
    np.set_printoptions(precision=4, suppress=True)
    print(a)
    print(norm0)
    print(min0)
    print(range0)


def datingClassTest():
    testRatio = 0.1
    dataset, labels = datingFile2Matrix('data/datingTestSet2.txt')
    # 由于前2个特征才有分类能力，所以只取前2个特征值
    dataset=dataset[:, 0:2]
    normMat, rangeValues, minValues = autoNorm(dataset)
    total = dataset.shape[0]
    testNum = int(total*testRatio)
    k = 5
    errorCount = 0
    smallDataSet = normMat[testNum:total, :]
    smallLabel = labels[testNum:total]
    fig = plt.figure()
    axis = fig.add_subplot(1,1,1)
    axis.scatter(smallDataSet[:,0],smallDataSet[:,1],20,smallLabel)
    plt.show()
    for i in range(testNum):
        sample = normMat[i, :]
        label = labels[i]
        y = knn(sample, smallDataSet, smallLabel, k)
        print("%d.\t real is %d, got is %d" % (i, label, y))
        if y != label:
            errorCount += 1
    print("error rate is: %f" % (errorCount/(1.0*testNum)))


def dating_knn():
    like = ['not at all', 'in small dose', 'in large dose']
    game = float(input('percent of playing video game time?'))
    miles = float(input('flying miles per year?'))
    icecream = float(input('ice cream consumed per year?'))
    dataset, labels = datingFile2Matrix('data/datingTestSet2.txt')
    # 由于前2个特征才有分类能力，所以只取前2个特征值
    # dataset = dataset[:, 0:2]
    normMat, rangeValues, minValues = autoNorm(dataset)
    cls = knn((np.array([miles, game, icecream]) - minValues)/rangeValues, normMat, labels, 5 )
    cls2 = like[cls - 1]
    print('you will probably like this person: ', cls2)


# 以下为knn手写数字识别
def knn_digit():
    path = './data/digits/trainingDigits/'
    files = os.listdir(path)
    traningSize = len(files)
    dataset = np.zeros((traningSize, 1024))
    labels = np.empty(traningSize)
    for i in range(traningSize):
        dataset[i, :], labels[i] = digitText2matrix(path + files[i])
    path = './data/digits/testDigits/'
    files = os.listdir(path)
    testSize = len(files)
    error = 0
    for i in range(testSize):
        sample, label = digitText2matrix(path+files[i])
        cls = knn(sample, dataset, labels, 5)
        if label != cls:
            print('sample %s:\t should be %d got %d' % (files[i], label, cls))
            error += 1
    print('error rate is %f' % (error*1.0/testSize))
    return 0


def digitText2matrix(file):
    fr = open(file)
    lines = fr.readlines()
    fr.close()
    mat = np.zeros(1024)
    for row in range(32):
        line = lines[row]
        for column in range(32):
            mat[row*32 + column] = int(line[column])
    label = os.path.split(file)[1].split('_')[0]
    return mat, int(label)


def plotDigit(file):
    """
    根据文件内容，用matplot显示一个数字
    :param file:
    :return:
    """
    mat, label = digitText2matrix(file)
    mat2 = np.zeros((1024, 3))
    for i in range(32):
        for j in range(32):
            index = i * 32 + j
            mat2[index, 0] = 32 - i
            mat2[index, 1] = j
            mat2[index, 2] = mat[index]
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    y = mat2[:,0]
    x = mat2[:,1]
    v = mat2[:,2]
    axis.scatter(x, y, v+1, v)
    plt.show()


def testPlostDigit():
    path = './data/digits/trainingDigits/'
    files = os.listdir(path)
    # np.random.seed(1234)
    for i in range(5):
        plotDigit(path+files[400+i])


if __name__ == '__main__':
    # datingDataFigure()
    # datingClassTest()
    # dating_knn()
    # knn_digit()
    testPlostDigit()
    input('press enter to exit ...')


