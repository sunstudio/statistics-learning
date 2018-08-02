
def operateOnArray(A, B, f):
    """对两个数组A和B按元素进行函数运算"""
    try:
        return [operateOnArray(a, b, f) for a, b in zip(A, B)]
    except TypeError as e:
        # Not iterable
        return f(A, B)


def operateOnVector(A, f):
    """对向量A按元素执行函数"""
    try:
        return [operateOnVector(a, f) for a, in A]
    except TypeError as e:
        # Not iterable
        return f(A)

