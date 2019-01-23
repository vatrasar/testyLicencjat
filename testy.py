import pickle
import numpy as np
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
D = 80 * 80 # input dimensionality: 80x80 grid

# model = {}
# model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
# model['W2'] = np.random.randn(H) / np.sqrt(H)
binary_file = open('my_pickled_mary.bin',mode='wb')
my_pickled_mary = pickle.dump(model, binary_file)
binary_file.close()
