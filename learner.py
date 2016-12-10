import theano
from theano import tensor as T
import numpy as np
import lasagne
from replay_memory import replay_memory
import cPickle

west = 2
east = 3
north = 4
south = 5
num_channels = 6
boardsize = 13
padding = 2
input_size = boardsize+2*padding
neighbor_patterns = ((-1,0), (0,-1), (-1,1), (0,1), (1,0), (1,-1))
input_shape = (num_channels,input_size,input_size)

class Learner:
    def __init__(self, loadfile = None, gamma = 1, alpha = 0.001, rho = 0.9, epsilon = 1e-6):
        self.mem = replay_memory(100000, input_shape)
        self.gamma = gamma

        #Create Input Variables
        state = T.tensor3('state')
        action = T.fvector('action')
        state_batch = T.tensor4('state_batch')
        action_batch = T.matrix('action_batch')

    def update_memory(self, state1, action, reward, state2, terminal):
        state1 = np.asarray(state1, dtype = theano.config.floatX).reshape(input_shape)
        state2 = np.asarray(state2, dtype = theano.config.floatX).reshape(input_shape)
        action = np.asarray(action, dtype = theano.config.floatX)
        self.mem.add_entry(state1, action, reward, state2, terminal)

    def learn(self, batch_size):
        #do nothing if we don't yet have enough entries in memory for a full batch
        if(self.mem.size < batch_size):
            return
        states1, actions, rewards, states2, terminals = self.mem.sample_batch(batch_size)
        targets = np.zeros(rewards.size).astype(theano.config.floatX)
        targets[terminals==0] = rewards[terminals==0]+self.gamma*self._evaluate_actions(states2, self._select_actions(states2))[terminals==0]
        targets[terminals==1] = rewards[terminals==1]

        self._update_Q(states1, actions, targets)
        self._update_P(states1)

    def save(self, savefile = 'learner.save'):
        params = lasagne.layers.get_all_param_values(self.output)
        data = {'params':p_params, 'mem':self.mem}
        with file(savefile, 'wb') as f:
            cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)