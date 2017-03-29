import theano
import numpy as np
import random as pr
from replay_memory import replay_memory
from inputFormat import *


def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

#TODO: Debug, figure out how to save and load rms_prop state along with any other needed info
class Learner:
    def __init__(self, 
        loadfile = None, 
        alpha = 0.001, 
        rho = 0.9, 
        epsilon = 1e-6, 
        mem_size = 100000,
        boardsize = 7):
        input_size = boardsize+2*padding
        input_shape = (num_channels,input_size,input_size)
        self.mem = replay_memory(mem_size, input_shape)

    def update_memory(self, state1, action, state2, terminal):
        self.mem.add_entry(state1, action, state2, terminal)

    def learn(self, batch_size):
        return 0

    def mentor(self, states, Pws, Qsigmas):
        return 0

    def exploration_policy(self, state, win_cutoff=0.0001, pruned = []):
        played = np.logical_or(state[white,padding:-padding,padding:-padding], state[black,padding:-padding,padding:-padding]).flatten()
        state = np.asarray(state, dtype=theano.config.floatX)
        action = np.random.choice(np.where(played==0)[0])
        return action, np.zeros(played.shape)

    def get_memory(self):
        return self.mem

    def optimization_policy(self, state):
        played = np.logical_or(state[white,padding:-padding,padding:-padding], state[black,padding:-padding,padding:-padding]).flatten()
        state = np.asarray(state, dtype=theano.config.floatX)
        action = np.random.choice(np.where(played==0)[0])
        return action

    def save(self, savefile = 'learner.save'):
        return