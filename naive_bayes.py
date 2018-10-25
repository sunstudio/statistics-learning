# machine learning in action
# classifying with probability theory: naive bayes
import numpy as np

def loadSimpleDataSet():
    # a simple data set, containing some posts
    postingList = [
        'my dog has flea problems, help please',
        'maybe not take him to dog park stupid',
        'my dalmation is so cute I love him',
        'stop posting stupid worthless garbage',
        'mr licks ate my steak how to stop him',
        'quit buying worthless dog food stupid'
    ]
    for i in range(len(postingList)):
        postingList[i] = postingList[i].split()
    classVec = [0,1,0,1,0,1]        # 1 is abusive, 0 not
    return postingList, classVec


def createVocabularyList(dataSet):
    # create vocabulary list from posts
    vocabulary = set()
    for sample in dataSet:
        vocabulary |= set(sample)
    words = list(vocabulary)
    # l.sort()
    return words


def wordsToVec(vocabulary, words):
    # convert a post to vocabulary vector
    result = np.zeros(len(vocabulary))
    for word in words:
        if word in vocabulary:
            result[vocabulary.index(word)] = 1
        else:
            print(f'word {word} does not in vocabulary')
    return result


def trainNB0(dataSet, labels):
    """
    training a data set and return the prior and conditional probability.
    can only be used with 2 categories text classification
    :param dataSet:training samples
    :param labels: corresponding class for each sample
    :return: the prior and conditional probability
    """
    sampleNum = len(dataSet)
    wordsNum = len(dataSet[0])
    prior = sum(labels) / sampleNum
    probability0 = np.ones(wordsNum)
    probability1 = np.ones(wordsNum)
    total0 = wordsNum
    total1 = wordsNum
    matrix = np.array(dataSet)
    for i in range(sampleNum):
        if labels[i] == 0:
            # abusive
            total1 += sum(matrix[i])
            probability0 += matrix[i]
        else:
            total0 += sum(matrix[i])
            probability1 += matrix[i]
    probability0 = np.log(probability0 / total0)
    probability1 = np.log(probability1 / total1)
    return probability0, probability1, prior


def classifyNB(vector, probality0, probality1, prior):
    p0 = sum(vector * probality0) + np.log(1 - prior)
    p1 = sum(vector * probality1) + np.log(prior)
    if p1 > p0:
        return 1
    else:
        return 0


def test():
    data, labels = loadSimpleDataSet()
    print('data', data)
    print('labels', labels)
    vocabulary = createVocabularyList(data)
    # print('vocabulary', vocabulary)
    wordVec = wordsToVec(vocabulary, data[0])
    # print('word vector', wordVec)
    wordVecList = [wordsToVec(vocabulary, words) for words in data]
    p0, p1, prior = trainNB0(wordVecList, labels)
    print('p0', p0)
    print('p1', p1)
    # print('prior', prior)
    sample = ['love','my','dog']
    vector = wordsToVec(vocabulary, sample)
    c1 = classifyNB(vector,p0,p1,prior)
    print(c1)
    sample = ['my', 'stupid', 'dog']
    vector = wordsToVec(vocabulary, sample)
    c1 = classifyNB(vector,p0,p1,prior)
    print(c1)


# following is spam email classification

def textParse(text):
    import re
    words = re.split(r'\W', text)
    return [word.lower() for word in words if len(word) > 2]


def spamTest():
    import os
    import os.path
    import random
    dataSet = []
    labels = []
    dirs = ['.\\data\\email\\ham\\', '.\\data\\email\\spam\\']
    for i in range(2):
        directory = dirs[i]
        files = os.listdir(directory)
        label = 1 if directory.find('spam') > 0 else 0
        for file in files:
            content = readFile(directory + file)
            words = textParse(content)
            dataSet.append(words)
            labels.append(label)
    vocabulary = createVocabularyList(dataSet)
    matrix = []
    for sample in dataSet:
        vector = wordsToVec(vocabulary, sample)
        matrix.append(vector)
    trainingIndex = list(range(len(matrix)))
    trainingLabels = []
    testIndex = []
    # randomly choose 10 samples as test set
    for i in range(10):
        index = random.choice(trainingIndex)
        testIndex.append(index)
        trainingIndex.remove(index)
    for i in trainingIndex:
        trainingLabels.append(labels[i])
    trainingSet = []
    for i in trainingIndex:
        trainingSet.append(matrix[i])
    p0, p1, prior = trainNB0(trainingSet, trainingLabels)
    errorCount = 0
    for i in testIndex:
        sample = matrix[i]
        prediction = classifyNB(sample, p0, p1, prior)
        if prediction != labels[i]:
            errorCount += 1
    print(f'error count is : {errorCount} and error rate is {errorCount/len(testIndex)}')


def readFile(file):
    f = open(file, 'r')
    content = f.read()
    f.close()
    return content


# test()
spamTest()
