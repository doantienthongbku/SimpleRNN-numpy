import numpy as np
from rnn import RNN

"""
matrix encoding by one-hot coding

'h': [1, 0, 0, 0]
'e': [0, 1, 0, 0]
'l': [0, 0, 1, 0]
'o': [0, 0, 0, 1]
"""

train_dataset = np.array([[1, 0, 0, 0],     # h
                          [0, 1, 0, 0],     # e
                          [0, 0, 1, 0],     # l
                          [0, 0, 1, 0]])    # l

target = np.array([[0, 1, 0, 0],    # e
                   [0, 0, 1, 0],    # l
                   [0, 0, 1, 0],    # l
                   [0, 0, 0, 1]])   # o

model = RNN()
model.fit(train_dataset, target, epochs=10, lr=0.001, print_cost=True)
