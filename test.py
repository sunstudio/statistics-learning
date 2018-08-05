import numpy as np


def axis_test():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print('a is: ', a)
    b = a.max(0)
    print('a.max(0) is:', b, b.shape)
    c = a.max(1)
    print('a.max(1) is:', c, c.shape)
    d = np.tile(b, (2, 1))
    print(d)
    e = np.tile(c, (2, 1))
    print(e)


def rangeTest():
    a = np.array([[1, 2, 3], [4, 5, 6],[7,8,9]])
    b = a[:, 2:2]
    print(b)


if __name__ == '__main__':
    rangeTest()
