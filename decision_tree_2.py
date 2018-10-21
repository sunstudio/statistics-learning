import math
import operator
import timeit
import matplotlib.pyplot as plt


class TreeNode:
    def __init__(self):
        self.level = 0
        self.feature = ''
        self.children = {}
        self.leafNumber = 0
        self.depth = 0

    def setLeaf(self, feature, level):
        self.feature = feature
        self.level = level

    def isLeaf(self):
        return len(self.children) == 0

    def __str__(self):
        s = '{'
        s += self.feature + "("+str(self.level)+"):"
        for child in self.children.keys():
            s += str(child) + ":"
            s += str(self.children[child])
        s += "}"
        return s

    def getLeafNumber(self):
        """
        get the number of leaves of a tree
        :param tree: the decision tree
        :return:
        """
        self.leafNumber = 0
        if self.isLeaf():
            self.leafNumber = 1
            return
        for child in self.children.values():
            child.getLeafNumber()
            self.leafNumber += child.leafNumber

    def getDepth(self):
        """
        get the depth of a decision tree
        :param tree:
        :return:
        """
        if self.isLeaf():
            return self.level
        for child in self.children.values():
            child.getDepth()
            if child.depth > self.depth:
                self.depth = child.depth

def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


# decision tree
def calculate_entropy(dataset:list):
    """
    calculate the entropy of a data set
    :param dataset: data set ( with the class at the last column)
    :return: entropy
    """
    number = len(dataset)
    label_count = {}
    entropy = 0
    for sample in dataset:
        label = sample[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1
    for label in label_count:
        probability = label_count[label] / number
        entropy -= probability*math.log(probability, 2)
    return entropy


def split_dataset(dataset: list, feature: int, value):
    """
    split the dataset , find the subset that has the specific value at the feature
    :param dataset: 数据集, each row is a sample
    :param feature: 指定特征
    :param value: 指定特征的值
    :return: 符合条件后的数据集（指定特征已经删除）
    """
    result = []
    for sample in dataset:
        if sample[feature] == value:
            temp = sample[:feature]
            temp.extend(sample[feature+1:])
            result.append(temp)
    return result


def choose_best_feature(dataset: list):
    """
    choose the best feature to split the data set (the feature make the most decrease of entropy)
    :param dataset:
    :return: the best feature
    """
    feature_num = len(dataset[0]) - 1
    best_feature = -1
    most_entory_decrease = 0
    original_entropy = calculate_entropy(dataset)
    for feature in range(feature_num):
        feature_values = set(dataset[feature])
        temp_entropy = 0
        for v in feature_values:
            sub = split_dataset(dataset, feature, v)
            temp_entropy += len(sub)/len(dataset)*calculate_entropy(sub)
        if original_entropy - temp_entropy > most_entory_decrease:
            best_feature = feature
            most_entory_decrease = original_entropy - temp_entropy
    return best_feature


def majority_vote(class_list: list):
    """
    majority vote
    :param class_list:
    :return: the majority class
    """
    class_vote = {}
    for c in class_list:
        if c not in class_vote.keys():
            class_vote[c] = 0
        class_vote[c] += 1
    sorted_list = sorted(class_vote, operator.itemgetter(1), True)
    return sorted_list[0][0]


def create_tree(dataset: list, labels: list, level):
    """
    create decision tree
    :param dataset:
    :param labels: feature names
    :return: the created decision tree
    """
    labels2 = labels[::]
    class_list = [sample[-1] for sample in dataset]
    node = TreeNode()
    if len(class_list) == class_list.count(class_list[0]):
        temp = TreeNode()
        temp.setLeaf(class_list[0], level)
        return temp
    if len(dataset[0]) == 1:
        temp = TreeNode()
        temp.setLeaf(majority_vote(class_list), level)
        return temp
    best = choose_best_feature(dataset)
    best_label = labels[best]
    node.level = level
    node.feature = best_label
    values = set([sample[best] for sample in dataset])
    labels2.remove(best_label)
    for v in values:
        sub = split_dataset(dataset, best, v)
        node.children[v] = create_tree(sub, labels2, level + 1)
    return node


def classify(tree, featureLables, sample):
    firstKey = tree.keys().__iter__().__next__()
    firstValue = tree[firstKey]
    featureIndex = featureLables.index(firstKey)
    for v in firstValue.keys():
        if v == sample[featureIndex]:
            if type(firstValue[v]).__name__ == 'dict':
                return classify(firstValue[v], featureLables, sample)
            else:
                return firstValue[v]


def saveTree(tree, fileName):
    import pickle
    stream = open(fileName, 'w')
    pickle.dump(tree, stream)
    stream.close()


def loadTree(fileName):
    import pickle
    stream = open(fileName, 'r')
    tree = pickle.load(stream)
    stream.close()
    return tree


def plotTree(tree):
    plt.figure(1)
    plt.clf()
    axis1 = plt.subplot(111)
    tree.getLeafNumber()
    tree.getDepth()
    plotSubTree(axis1, tree, (0.5, 0.95, 0.9, 0.9), None, (0.9/tree.leafNumber, 0.9/tree.depth))
    plt.show()


def plotSubTree(axis, node, rect, parentPosition=None, step=(0.1, 0.1)):
    """
    draw a decision tree
    :param node: a node of a decision tree
    :param rect: the rectangle(x,y,w,h) in which the tree plots, and (x,y) is top center of the rectangle
    :param parentPosition: the position(x,y) of the parent node
    :param step: the distance(x,y) between adjacent nodes in the tree
    :return: none
    """
    if type(node).__name__ == 'dict':
        firstKey = node.keys().__iter__().__next__()
        firstValue = node[firstKey]
        childrenNum = len(firstValue)
    else:
        firstKey = node
        firstValue = node
    if parentPosition:
        axis.annotate(firstKey, xy=parentPosition, xytext=(rect[0],rect[1]),
                      arrowprops={'arrowstyle':'<-'}, bbox={'boxstyle':'round4'},
                      verticalalignment="top", horizontalalignment="center")
    else:
        axis.annotate(firstKey, xytext=(rect[0],rect[1]), xy=(rect[0],rect[1]),
                      bbox={'boxstyle':'round4'},
                      verticalalignment="top", horizontalalignment="center")
    if type(firstValue).__name__ != 'dict':
        return
    i = 0
    left = rect[0] - rect[3]/2
    totalLeafs = node.leafNumber
    for key in firstValue.keys():
        child = firstValue[key]
        leafs = child.leafNumber
        newRect = (left + step[0]*(i+leafs/2), rect[1]-step[1], step[0]*leafs, 0)
        plotSubTree(axis, firstValue[key], newRect, (rect[0],rect[1]), step)
        axis.text((rect[0]+newRect[0])/2, (rect[1]+newRect[1])/2, key)
        i += leafs



def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers':  {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers':  {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]


def test_tree():
    dataset, labels = create_dataset()
    print(dataset)
    print(labels)
    # entropy = calculate_entropy(dataset)
    # print("entropy :", entropy)
    # sub = split_dataset(dataset, 1, 1)
    # print('after split', sub)
    # best = choose_best_feature(dataset)
    # print('best feature:', best)
    tree = create_tree(dataset, labels, 1)
    print('tree')
    print(tree)
    plotTree(tree)
    """
    sample = [1,0]
    c1 = classify(tree, labels, sample)
    print(f'the class of {sample} is {c1}')
    sample = [1,1]
    c1 = classify(tree, labels, sample)
    print(f'the class of {sample} is {c1}')
    """


def testContactLens():
    stream = open('./data/lenses.txt', 'r')
    dataSet = [line.strip().split('\t') for line in stream.readlines()]
    print(dataSet)
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    tree = create_tree(dataSet, labels, 1)
    print(tree)
    # plotTree(tree)


# a = timeit.timeit(stmt='test_tree()', setup='from __main__ import test_tree', number=1)
# print(a)
# testContactLens()
test_tree()

