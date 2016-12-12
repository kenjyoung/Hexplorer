import theano
from theano import tensor as T
import numpy as np
import lasagne
from replay_memory import replay_memory
from layers import HexConvLayer
import cPickle

white = 0
black = 1
west = 2
east = 3
north = 4
south = 5
num_channels = 6
padding = 2
boardsize = 13
neighbor_patterns = ((-1,0), (0,-1), (-1,1), (0,1), (1,0), (1,-1))

#TODO: Debug, figure out how to save and load rms_prop state along with any other needed info

class Learner:
    def __init__(self, 
        loadfile = None, 
        alpha = 0.001, 
        rho = 0.9, 
        epsilon = 1e-6, 
        mem_size = 100000,
        boardsize = 13):

        input_size = boardsize+2*padding
        input_shape = (num_channels,input_size,input_size)
        self.mem = replay_memory(mem_size, input_shape)

        #Create Input Variables
        state = T.tensor3('state')
        state_batch = T.tensor4('state_batch')
        action_batch = T.matrix('action_batch')
        mentor_Pws = T.tensor3('mentor_Pws')
        mentor_Qsigmas = T.tensor3('mentor_Qsigmas')
        Pw_targets = T.fvector('Pw_targets')
        Qsigma_targets = T.fvector('Qsigma_targets')

        #Load from file if given
        if(loadfile != None):
            with file(loadfile, 'rb') as f:
                data = cPickle.load(f)
            params = data["params"]
            self.mem = data["mem"]

        self.layers = []

        #Initialize input layer
        l_in = lasagne.layers.InputLayer(
            shape=(None, input_shape[0],input_shape[1],input_shape[2]),
            input_var = state_batch
        )
        self.layers.append(l_in)

        #Initialize bottom radius 3 layer
        l_1 = HexConvLayer(
            incoming = l_in, 
            num_filters=128, 
            radius = 3, 
            nonlinearity = lasagne.nonlinearity.rectify, 
            W=lasagne.init.HeNormal(gain='relu'), 
            b=lasagne.init.Constant(0),
            padding = 1,
        )
        self.layers.append(l_1)

        #Initialize layers shared by Pw and Qsigma networks
        num_shared = 6
        for i in range(num_shared-1):
            layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=128, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearity.rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 1,
            )
            self.layers.append(layer)
        final_shared_layer = self.layers[-1]

        #Initialize layers unique to Pw network
        num_Pw = 6
        layer = HexConvLayer(
                incoming = final_shared_layer, 
                num_filters=128, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearity.rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 1,
            )
        self.layers.append(layer)
        for i in range(num_Pw-2):
            layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=128, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearity.rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 1,
            )
            self.layers.append(layer)
        Pw_output = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=128, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearity.sigmoid, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 1,
            )
        self.layers.append(Pw_output)

        #Initialize layers unique to Qsigma network
        num_Qsigma = 6
        layer = HexConvLayer(
                incoming = final_shared_layer, 
                num_filters=128, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearity.rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 1,
            )
        self.layers.append(layer)
        for i in range(num_Qsigma-2):
            layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=128, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearity.rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 1,
            )
            self.layers.append(layer)
        Qsigma_output = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=128, 
                radius = 2, 
                nonlinearity = lambda x: 2*lasagne.nonlinearities.sigmoid(x)-1, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 0,
            )
        self.layers.append(Qsigma_output)

        #If a loadfile is given, use saved parameter values
        if(loadfile != None):
            lasagne.layers.set_all_param_values(self.layers, params)

        #Build functions
        #===============

        #Compute played so we can set the outputs for all played cells to 0, which will enforce they
        #don't effect updates and evaluations
        played = 1-(1-state_batch[:,white,:,:])*(1-state_batch[:,black,:,:])

        #Build Pw evaluate functions
        self._evaluate_Pw = theano.function(
            [state],
            givens = {state_batch : state.dimshuffle('x',0,1,2)},
            outputs = Pw_output*(1-played)
            )
        self._evaluate_Pws = theano.function(
            [state_batch],
            outputs = Pw_output*(1-played)
        )

        #Build Qsigma evaluate functions
        self._evaluate_Qsigma = theano.function(
            [state],
            givens = {state_batch : state.dimshuffle('x',0,1,2)},
            outputs = Qsigma_output*(1-played)
            )
        self._evaluate_Qsigmas = theano.function(
            [state_batch],
            outputs = Qsigma_output*(1-played)
        )

        #Build functions to evaluate both Qsigma and Pw
        self._evaluate = theano.function(
            [state],
            givens = {state_batch : state.dimshuffle('x',0,1,2)},
            outputs = [Pw_output*(1-played), Qsigma_output*(1-played)]
            )
        self._evaluate_multi = theano.function(
            [state_batch],
            outputs = [Pw_output*(1-played), Qsigma_output*(1-played)]
        )

        #Build Pw update function
        Pw_loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(Pw_output.flatten()[T.arange(Pw_targets.shape[0]),action_batch], Pw_targets), mode='mean')
        Pw_params = lasagne.layers.get_all_params(Pw_output)
        Pw_updates = lasagne.update.rmsprop(Pw_loss, Pw_params, alpha, rho, epsilon)
        self._update_Pw = theano.function(
            [state_batch, action_batch, Pw_targets],
            updates = Pw_updates
        )

        #Build Qsigma update function
        Qsigma_loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(Qsigma_output.flatten()[T.arange(Qsigma_targets.shape[0]),action_batch], Qsigma_targets), mode='mean')
        Qsigma_params = lasagne.layers.get_all_params(Qsigma_output)
        Qsigma_updates = lasagne.update.rmsprop(Qsigma_loss, Qsigma_params, alpha, rho, epsilon)
        self._update_Qsigma = theano.function(
            [state_batch, action_batch, Qsigma_targets],
            updates = Qsigma_updates
        )

        #Build Pw mentor function
        Pw_loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(Pw_output.flatten(),mentor_Pws.flatten()))
        Pw_params = lasagne.layers.get_all_params(Pw_output)
        Pw_updates = lasagne.update.rmsprop(Pw_loss, Pw_params, alpha, rho, epsilon)
        self._mentor_Pw = theano.function(
            [state_batch, mentor_Pws],
            updates = Pw_updates
        )

        #Build Qsigma mentor function
        Qsigma_loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(Qsigma_output.flatten(),mentor_Qsigmas.flatten()))
        Qsigma_params = lasagne.layers.get_all_params(Qsigma_output)
        Qsigma_updates = lasagne.update.rmsprop(Qsigma_loss, Qsigma_params, alpha, rho, epsilon)
        self._mentor_Qsigma = theano.function(
            [state_batch, mentor_Qsigmas],
            updates = Qsigma_updates
        )

        #Build mentor function for both Pw and Q_sigma
        loss = Pw_loss + Qsigma_loss
        params = Pw_params + Qsigma_params
        updates = lasagne.update.rmsprop(loss, params, alpha, rho, epsilon)
        self._mentor = theano.function(
            [state_batch, mentor_Pws, mentor_Qsigmas],
            updates = updates
        )

    def update_memory(self, state1, action, state2, terminal):
        self.mem.add_entry(state1, action, state2, terminal)

    def learn(self, batch_size):
        #Do nothing if we don't yet have enough entries in memory for a full batch
        if(self.mem.size < batch_size):
            return
        states1, actions, states2, terminals = self.mem.sample_batch(batch_size)
        Pw, Qsigma = self._evaluate_multi(states2)
        joint = np.prod(1-Pw, axis=(1,2))

        #Update Pw network
        Pw_targets = np.zeros(terminals.size).astype(theano.config.floatX)
        Pw_targets[terminals==0] = joint[terminals==0]
        Pw_targets[terminals==1] = -1
        self._update_Pw(states1, actions, Pw_targets)

        #Update Qsigma network
        gamma = (joint/(1-Pw))**2
        Qsigma_targets = np.zeros(terminals.size).astype(theano.config.floatX)
        Qsigma_targets = Pw.flatten()[T.arange(batch_size),actions]**2-joint**2+np.max(gamma*Qsigma)
        self._update_Qsigma(states1, actions, Qsigma_targets)

    def mentor(self, states, Pws, Qsigmas):
        self._mentor(states, Pws, Qsigmas)

    def exploration_policy(self, state):
        Pw, Qsigma = self._evaluate(state)
        joint = np.prod(1-Pw, axis=(1,2))
        gamma = (joint/(1-Pw))**2
        action = np.argmax((gamma*Qsigma).flatten())
        return action

    def optimization_policy(self, state):
        Pw = self._evaluate_Pw(state).flatten()
        action = np.argmax(Pw.flatten())
        return action

    def save(self, savefile = 'learner.save'):
        params = lasagne.layers.get_all_param_values(self.layers)
        data = {'params':params, 'mem':self.mem}
        with file(savefile, 'wb') as f:
            cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)