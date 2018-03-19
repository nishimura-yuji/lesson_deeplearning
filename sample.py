import numpy as np
import matplotlib.pyplot as plt


# x = np.arange(0, 6, 0.1)
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0  # ok
    else:
        return 1  # NG


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 重みとバイアスだけがANDと違う!
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # 重みとバイアスだけがANDと違う!
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


#
# XOR(0, 0)  # 0 を出力
# XOR(1, 0)  # 1 を出力
# XOR(0, 1)  # 1 を出力
# XOR(1, 1)  # 0 を出力


def step_function(x):
    """

    >>> import numpy as np
    >>> x = np.array([-1.0, 1.0, 2.0])
    >>> x
    array([-1., 1., 2.])
    >>> y = x > 0
    >>> y
    array([False, True, True], dtype=bool)
    """
    # if x > 0:
    #     return 1
    # else:
    #     return 0
    y = x > 0
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y 軸の範囲を指定
plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y 軸の範囲を指定
plt.show()


def relu(x):
    return np.maximum(0, x)


###
# 多次元配列
###

