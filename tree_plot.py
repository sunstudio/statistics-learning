import matplotlib.pyplot as plt

# This module should be used with 'decision_tree' module.
# Code in this file is to show the structure of the decision tree created by the 'decision_tree' module.

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
    test()