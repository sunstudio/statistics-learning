import  math
import  operator


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['不需浮出水面', '有蹼']
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


def create_tree(dataset: list, labels: list):
    """
    create decision tree
    :param dataset:
    :param labels: feature names
    :return: the created decision tree
    """
    class_list = [sample[-1] for sample in dataset]
    if len(class_list) == class_list.count(class_list[0]):
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_vote(class_list)
    best = choose_best_feature(dataset)
    best_label = labels[best]
    tree = {best_label: {}}
    values = set([sample[best] for sample in dataset])
    labels.remove(best_label)
    for v in values:
        sub = split_dataset(dataset, best, v)
        tree[best_label][v] = create_tree(sub, labels[:])
    return tree


def test_dataset():
    dataset, labels = create_dataset()
    print(dataset)
    print(labels)
    entropy = calculate_entropy(dataset)
    print("entropy :", entropy)
    sub = split_dataset(dataset, 1, 1)
    print('after split', sub)
    best = choose_best_feature(dataset)
    print('best feature:', best)
    tree = create_tree(dataset, labels)
    print('tree')
    print(tree)


if __name__ == "__main__":
    test_dataset()



