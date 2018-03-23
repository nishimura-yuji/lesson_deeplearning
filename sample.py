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


#
# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)  # y 軸の範囲を指定
# plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#
#
# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)  # y 軸の範囲を指定
# plt.show()


def relu(x):
    return np.maximum(0, x)


###
# 多次元配列
###

##  ニューラルネットワークの行列の席

def nuralnetwork():
    X = np.array([1, 2])
    X_data = X.shape
    print(X_data)
    W = np.array([[1, 3, 5], [2, 4, 6]])
    print(W)
    Y = np.dot(X, W)
    print(Y)


# nuralnetwork()

## 多次元配列ニューラルネット

# X = np.array([1.0, 0.5])
# W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
# B1 = np.array([0.1, 0.2, 0.3])

# print(W1.shape)
# print(X.shape)
# print(B1.shape)
#
# A1 = np.dot(X, W1) + B1
#
# Z1 = sigmoid(A1)
#
# print(A1)
# print(Z1)
#
# W2 = np.array([[0.1, 0.4], [0.2, 0, 5], [0.3, 0.6]])
#
# B2 = np.array([0.1, 0.2])
#
# print(Z1.shape)
# print(W2.shape)
# print(B2.shape)
#
# A2 = np.dot(Z1, W2) + B2
# Z2 = sigmoid(A2)


def identity_function(x):
    return x

#
# W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
# B3 = np.array([0.1, 0.2])
#
# A3 = np.dot(Z2, W3) + B3

####
## 実装まとめ
####

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

# network = init_network()
# x = np.array([1.0, 0.5])
# y = forward(network, x)
# print(y) # [ 0.31682708  0.69627909]


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y