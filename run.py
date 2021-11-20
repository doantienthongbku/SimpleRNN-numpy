import numpy as np
from rnn import RNN

"""
matrix encoding by one-hot coding

'h': [1, 0, 0, 0]
'e': [0, 1, 0, 0]
'l': [0, 0, 1, 0]
'o': [0, 0, 0, 1]
"""


def create_dataset():
    train_dataset = {}
    target = {}

    train_dataset[0] = np.array([[1], [0], [0], [0]])
    train_dataset[1] = np.array([[0], [1], [0], [0]])
    train_dataset[2] = np.array([[0], [0], [1], [0]])
    train_dataset[3] = np.array([[0], [0], [1], [0]])

    target[0] = np.array([[0], [1], [0], [0]])
    target[1] = np.array([[0], [0], [1], [0]])
    target[2] = np.array([[0], [0], [1], [0]])
    target[3] = np.array([[0], [0], [0], [1]])

    return train_dataset, target


train_dataset, target = create_dataset()


model = RNN()
model.fit(train_dataset, target, epochs=100, lr=0.001, print_cost=True)
