
import numpy as np
import operator
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels


def classify0(sample,dataSet,labels,k):
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
    return  sortedClassCount[0][0]


def simpleKnnTest():
    group,labels = createDataSet()
    r = classify0([0, 0], group, labels, 3)
    print('class is ', r)


def file2Matrix(filename):
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


def testFile2Matrix():
    a, b = file2Matrix('data/datingTestSet2.txt')
    print(a[0:10])
    print(b[0:10])


def datingDataFigure():
    a, b = file2Matrix('data/datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(a[:, 0], a[:, 1], 30, np.array(b), '.')
    plt.show()


if __name__ == '__main__':
    datingDataFigure()