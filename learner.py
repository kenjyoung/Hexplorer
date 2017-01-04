import theano
from theano import tensor as T
import numpy as np
import lasagne
from replay_memory import replay_memory
from layers import HexConvLayer
from inputFormat import *
import pickle

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
        action_batch = T.ivector('action_batch')
        mentor_Pws = T.tensor3('mentor_Pws')
        mentor_Qsigmas = T.tensor3('mentor_Qsigmas')
        Pw_targets = T.fvector('Pw_targets')
        Qsigma_targets = T.fvector('Qsigma_targets')

        #Load from file if given
        if(loadfile != None):
            with open(loadfile, 'rb') as f:
                data = pickle.load(f)
            params = data["params"]
            self.mem = data["mem"]

        self.layers = []
        num_filters = 128
        num_shared = 6
        num_Pw = 6
        num_Qsigma = 6

        #Initialize input layer
        l_in = lasagne.layers.InputLayer(
            shape=(None, input_shape[0],input_shape[1],input_shape[2]),
            input_var = state_batch
        )
        self.layers.append(l_in)

        #Initialize bottom radius 3 layer
        l_1 = HexConvLayer(
            incoming = l_in, 
            num_filters=num_filters, 
            radius = 3, 
            nonlinearity = lasagne.nonlinearities.rectify, 
            W=lasagne.init.HeNormal(gain='relu'), 
            b=lasagne.init.Constant(0),
            padding = 1,
        )
        self.layers.append(l_1)

        #Initialize layers shared by Pw and Qsigma networks
        for i in range(num_shared-1):
            layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=num_filters, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearities.rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 1,
            )
            self.layers.append(layer)
        final_shared_layer = self.layers[-1]

        #Initialize layers unique to Pw network
        layer = HexConvLayer(
                incoming = final_shared_layer, 
                num_filters=num_filters, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearities.rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 1,
            )
        self.layers.append(layer)
        for i in range(num_Pw-2):
            layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=num_filters, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearities.rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 1,
            )
            self.layers.append(layer)
        Pw_output_layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=1, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearities.sigmoid, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 0,
            )
        self.layers.append(Pw_output_layer)
        Pw_output = lasagne.layers.get_output(Pw_output_layer)

        #Initialize layers unique to Qsigma network
        layer = HexConvLayer(
                incoming = final_shared_layer, 
                num_filters=num_filters, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearities.rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 1,
            )
        self.layers.append(layer)
        for i in range(num_Qsigma-2):
            layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=num_filters, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearities.rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 1,
            )
            self.layers.append(layer)
        Qsigma_output_layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=1, 
                radius = 2, 
                nonlinearity = lambda x: 2*lasagne.nonlinearities.sigmoid(x)-1, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                padding = 0,
            )
        self.layers.append(Qsigma_output_layer)
        Qsigma_output = lasagne.layers.get_output(Qsigma_output_layer)

        #If a loadfile is given, use saved parameter values
        if(loadfile != None):
            lasagne.layers.set_all_param_values(self.layers, params)

        #Build functions
        #===============

        #Compute played so we can set the outputs for all played cells to 0, which will enforce they
        #don't effect updates and evaluations
        played = 1-(1-state_batch[:,white,padding:-padding,padding:-padding])*(1-state_batch[:,black,padding:-padding,padding:-padding])

        #Build Pw evaluate functions
        self._evaluate_Pw = theano.function(
            [state],
            givens = {state_batch : state.dimshuffle('x',0,1,2)},
            outputs = (Pw_output*(1-played)).flatten()
        )
        self._evaluate_Pws = theano.function(
            [state_batch],
            outputs = (Pw_output*(1-played)).flatten(2)
        )

        #Build Qsigma evaluate functions
        self._evaluate_Qsigma = theano.function(
            [state],
            givens = {state_batch : state.dimshuffle('x',0,1,2)},
            outputs = (Qsigma_output*(1-played)).flatten()
        )
        self._evaluate_Qsigmas = theano.function(
            [state_batch],
            outputs = (Qsigma_output*(1-played)).flatten(2)
        )

        #Build functions to evaluate both Qsigma and Pw
        self._evaluate = theano.function(
            [state],
            givens = {state_batch : state.dimshuffle('x',0,1,2)},
            outputs = [(Pw_output*(1-played)).flatten(), (Qsigma_output*(1-played)).flatten()]
        )
        self._evaluate_multi = theano.function(
            [state_batch],
            outputs = [(Pw_output*(1-played).dimshuffle(0,'x',1,2)).flatten(2), (Qsigma_output*(1-played).dimshuffle(0,'x',1,2)).flatten(2)]
        )

        #Build Pw update function
        Pw_loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(Pw_output.flatten(2)[T.arange(Pw_targets.shape[0]),action_batch], Pw_targets), mode='mean')
        Pw_params = lasagne.layers.get_all_params(Pw_output_layer)
        Pw_updates = lasagne.updates.rmsprop(Pw_loss, Pw_params, alpha, rho, epsilon)
        self._update_Pw = theano.function(
            [state_batch, action_batch, Pw_targets],
            updates = Pw_updates,
            outputs = Pw_loss
        )

        #Build Qsigma update function
        Qsigma_loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(Qsigma_output.flatten(2)[T.arange(Qsigma_targets.shape[0]),action_batch], Qsigma_targets), mode='mean')
        Qsigma_params = lasagne.layers.get_all_params(Qsigma_output_layer)
        Qsigma_updates = lasagne.updates.rmsprop(Qsigma_loss, Qsigma_params, alpha, rho, epsilon)
        self._update_Qsigma = theano.function(
            [state_batch, action_batch, Qsigma_targets],
            updates = Qsigma_updates,
            outputs = Qsigma_loss
        )

        #Build update function for both Pw and Qsigma
        loss = Pw_loss + Qsigma_loss
        params = Pw_params + Qsigma_params
        updates = lasagne.updates.rmsprop(loss, params, alpha, rho, epsilon)
        self._update = theano.function(
            [state_batch, action_batch, Pw_targets, Qsigma_targets],
            updates = updates,
            outputs = [Pw_loss, Qsigma_loss]
        )

        #Build Pw mentor function
        Pw_mentor_loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(Pw_output.flatten(),mentor_Pws.flatten()))
        Pw_mentor_updates = lasagne.updates.rmsprop(Pw_mentor_loss, Pw_params, alpha, rho, epsilon)
        self._mentor_Pw = theano.function(
            [state_batch, mentor_Pws],
            updates = Pw_mentor_updates,
            outputs = Pw_mentor_loss
        )

        #Build Qsigma mentor function
        Qsigma_mentor_loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(Qsigma_output.flatten(),mentor_Qsigmas.flatten()))
        Qsigma_mentor_updates = lasagne.updates.rmsprop(Qsigma_mentor_loss, Qsigma_params, alpha, rho, epsilon)
        self._mentor_Qsigma = theano.function(
            [state_batch, mentor_Qsigmas],
            updates = Qsigma_mentor_updates,
            outputs = Qsigma_mentor_loss
        )

        #Build mentor function for both Pw and Qsigma
        loss = Pw_mentor_loss + Qsigma_mentor_loss
        params = Pw_params + Qsigma_params
        updates = lasagne.updates.rmsprop(loss, params, alpha, rho, epsilon)
        self._mentor = theano.function(
            [state_batch, mentor_Pws, mentor_Qsigmas],
            updates = updates,
            outputs = [Pw_mentor_loss, Qsigma_mentor_loss]
        )

    def update_memory(self, state1, action, state2, terminal):
        self.mem.add_entry(state1, action, state2, terminal)

    def learn(self, batch_size):
        #Do nothing if we don't yet have enough entries in memory for a full batch
        if(self.mem.size < batch_size):
            return
        states1, actions, states2, terminals = self.mem.sample_batch(batch_size)

        Pw, Qsigma = self._evaluate_multi(states2)
        #add a cap on the lowest possible value of losing probability
        Pl =  np.maximum(1-Pw,0.00001)
        joint = np.prod(Pl, axis=1)

        #Update networks
        Pw_targets = np.zeros(terminals.size).astype(theano.config.floatX)
        Pw_targets[terminals==0] = joint[terminals==0]
        Pw_targets[terminals==1] = 1
        gamma = (joint[...,np.newaxis]/Pl)**2
        Qsigma_targets = np.zeros(terminals.size).astype(theano.config.floatX)
        Qsigma_targets = Pw[np.arange(batch_size),actions]**2-joint**2+np.max(gamma*Qsigma, axis=1)
        return self._update(states1, actions, Pw_targets, Qsigma_targets)

    def mentor(self, states, Pws, Qsigmas):
        states = np.asarray(states, dtype=theano.config.floatX)
        Pws = np.asarray(Pws, dtype=theano.config.floatX)
        Qsigmas = np.asarray(Qsigmas, dtype=theano.config.floatX)
        return self._mentor(states, Pws, Qsigmas)

    def exploration_policy(self, state, win_cutoff=0.0001):
        played = np.logical_or(state[white,padding:-padding,padding:-padding], state[black,padding:-padding,padding:-padding]).flatten()
        state = np.asarray(state, dtype=theano.config.floatX)
        Pw, Qsigma = self._evaluate(state)
        #add a cap on the lowest possible value of losing probability
        Pl =  np.maximum(1-Pw,0.00001)
        joint = np.prod(Pl)
        #if losing probability is sufficiently small just play the best move
        if joint < win_cutoff:
        	values = np.copy(Pw)
	        #never select played values
	        values[played]=-2
	        action = np.argmax(values)
	        return action, Pw, Qsigma

        gamma = (joint/Pl)**2
        values = gamma*Qsigma
        #never select played values
        values[played]=-2
        action = np.argmax(values)
        return action, Pw, Qsigma

    def optimization_policy(self, state):
        played = np.logical_or(state[white,padding:-padding,padding:-padding], state[black,padding:-padding,padding:-padding]).flatten()
        state = np.asarray(state, dtype=theano.config.floatX)
        Pw = self._evaluate_Pw(state)
        values = Pw
        #never select played values
        values[played]=-2
        action = np.argmax(values)
        return action

    def win_prob(self, state):
        played = np.logical_or(state[white,padding:-padding,padding:-padding], state[black,padding:-padding,padding:-padding]).flatten()
        state = np.asarray(state, dtype=theano.config.floatX)
        Pw = self._evaluate_Pw(state)
        values = Pw
        return values

    def win_prob_and_exp(self, state):
        played = np.logical_or(state[white,padding:-padding,padding:-padding], state[black,padding:-padding,padding:-padding]).flatten()
        state = np.asarray(state, dtype=theano.config.floatX)
        Pw, Qsigma = self._evaluate(state)
        Pw_values = Pw
        Qsigma_values = Qsigma
        return Pw_values, Qsigma_values


    def save(self, savefile = 'learner.save'):
        params = lasagne.layers.get_all_param_values(self.layers)
        data = {'params':params, 'mem':self.mem}
        with open(savefile, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)