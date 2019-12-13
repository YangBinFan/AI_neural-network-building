import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5, 0.7], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]])
    network['b1'] = np.array([0.1, 0.2, 0.3, 0.4])
    network['W2'] = np.array([[0.1, 0.4, 0.7], [0.2, 0.5, 0.8], [0.3, 0.6, 0.9], [0.4, 0.5, 0.6]])
    network['b2'] = np.array([0.1, 0.2, 0.3])
    network['W3'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.6, 0.9]])
    network['b3'] = np.array([0.1, 0.2, 0.3])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5, 1.5])
y = forward(network, x)

print(y)




