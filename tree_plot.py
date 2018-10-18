import matplotlib.pyplot as plt
import decision_tree

# This module should be used with 'decision_tree' module.
# Code in this file is to show the structure of the decision tree created by the 'decision_tree' module.
decision_node = {'boxstyle':'sawtooth'}
leaf_node={'boxstyle':'round4','fc':'0.8'}
arrow = {'arrowstyle':'->'}


def get_leaf_number(tree):
    """
    get the number of leaves of a tree
    :param tree: the decision tree
    :return:
    """
    num = 0
    firstValue = tree.values().__iter__().__next__()
    for key in firstValue.keys():
        if type(firstValue[key]).__name__ == 'dict':
            num += get_leaf_number(firstValue[key])
        else:
            num += 1
    return num


def get_tree_depth(tree):
    """
    get the depth of a decision tree
    :param tree:
    :return:
    """
    depth = 0
    firstValue = tree.values().__iter__().__next__()
    # check every child tree, get max depth between them
    for key in firstValue.keys():
        if type(firstValue[key]).__name__ == 'dict':
            temp = get_tree_depth(firstValue[key])
            if temp > depth:
                depth = temp
    return depth + 1


def plotTree():
    plt.figure(1)
    plt.clf()
    axis1 = plt.subplot(111)
    dataset, labels = decision_tree.create_dataset()
    tree = decision_tree.create_tree(dataset, labels)
    print(tree)
    plotSubTree(axis1, tree, (0.5, 0.9), None, (0.2, 0.3))
    plt.show()


def plotSubTree(axis, node, position, parentPosition=None, step=(0.1, 0.1)):
    """
    draw a decision tree
    :param node: a node of a decision tree
    :param position: the position(x,y) of the root of the tree
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
        axis.annotate(firstKey, xy=parentPosition, xytext=position,
                      arrowprops={'arrowstyle':'<-'}, bbox={'boxstyle':'round4'},
                      verticalalignment="top", horizontalalignment="center")
    else:
        axis.annotate(firstKey, xytext=position, xy=position,
                      bbox={'boxstyle':'round4'},
                      verticalalignment="top", horizontalalignment="center")
    if type(firstValue).__name__ != 'dict':
        return
    i = 0
    for key in firstValue.keys():
        newPos = ((i - childrenNum/2)*step[0]+position[0], position[1]-step[1])
        plotSubTree(axis, firstValue[key], newPos, position, step)
        axis.text( (position[0]+newPos[0])/2, (position[1]+newPos[1])/2, key )
        i += 1



def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers':  {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers':  {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]


def test():
    tree = retrieveTree(0)
    n = get_leaf_number(tree)
    print(f'leaf number:{n}')
    d = get_tree_depth(tree)
    print(f'depth:{d}')


if __name__=='__main__':
    plotTree()