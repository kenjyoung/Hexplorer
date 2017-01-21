import theano
from theano import tensor as T
import numpy as np
import random as pr
import lasagne
from lasagne.regularization import regularize_layer_params, l2
from replay_memory import replay_memory
from layers import HexConvLayer
from inputFormat import *
import pickle
from collections import OrderedDict

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

def get_or_compute_grads(loss_or_grads, params):
    """
    Helper function returning a list of gradients.
    """
    if any(not isinstance(p, theano.compile.SharedVariable) for p in params):
        raise ValueError("params must contain shared variables only. If it "
                         "contains arbitrary parameter expressions, then "
                         "lasagne.utils.collect_shared_vars() may help you.")
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)

def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6, accu_vals = None):
    """
    Modified from lasagne version.
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    accu_list = []
    index = 0
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        if accu_vals is None:
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        else:
            accu =theano.shared(accu_vals[index],
                                 broadcastable=param.broadcastable)
        accu_list.append(accu)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))
        index += 1

    return updates, accu_list

#TODO: Debug, figure out how to save and load rms_prop state along with any other needed info
class Learner:
    def __init__(self, 
        loadfile = None, 
        alpha = 0.001, 
        rho = 0.9, 
        epsilon = 1e-6, 
        mem_size = 100000,
        boardsize = 5):
        input_size = boardsize+2*padding
        input_shape = (num_channels,input_size,input_size)

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
            opt_vals = data["opt"]
            self.mem = data["mem"]
        else:
            params = None
            opt_vals = None
            self.mem = replay_memory(mem_size, input_shape)

        self.opt_state = []
        self.layers = []
        num_filters = 32
        num_shared = 3
        num_Pw = 4
        num_Qsigma = 4

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
            nonlinearity = lasagne.nonlinearities.leaky_rectify, 
            W=lasagne.init.HeNormal(gain='relu'), 
            b=lasagne.init.Constant(0),
            pos_dep_bias = False,
            padding = 1,
        )
        self.layers.append(l_1)

        #Initialize layers shared by Pw and Qsigma networks
        for i in range(num_shared-1):
            layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=num_filters, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearities.leaky_rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                pos_dep_bias = False,
                padding = 1,
            )
            self.layers.append(layer)
        final_shared_layer = self.layers[-1]

        #Initialize layers unique to Pw network
        layer = HexConvLayer(
                incoming = final_shared_layer, 
                num_filters=num_filters, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearities.leaky_rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                pos_dep_bias = False,
                padding = 1,
            )
        self.layers.append(layer)
        for i in range(num_Pw-3):
            layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=num_filters, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearities.leaky_rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                pos_dep_bias = False,
                padding = 1,
            )
            self.layers.append(layer)
        layer = HexConvLayer(
            incoming = self.layers[-1], 
            num_filters=num_filters, 
            radius = 2, 
            nonlinearity = lasagne.nonlinearities.leaky_rectify, 
            W=lasagne.init.HeNormal(gain='relu'), 
            b=lasagne.init.Constant(0),
            pos_dep_bias = False,
            padding = 0,
        )
        self.layers.append(layer)
        Pw_output_layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=1, 
                radius = 1, 
                nonlinearity = lasagne.nonlinearities.sigmoid, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                pos_dep_bias = True,
                padding = 0,
            )
        self.layers.append(Pw_output_layer)
        Pw_output = lasagne.layers.get_output(Pw_output_layer)

        #Initialize layers unique to Qsigma network
        layer = HexConvLayer(
                incoming = final_shared_layer, 
                num_filters=num_filters, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearities.leaky_rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                pos_dep_bias = False,
                padding = 1,
            )
        self.layers.append(layer)
        for i in range(num_Qsigma-3):
            layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=num_filters, 
                radius = 2, 
                nonlinearity = lasagne.nonlinearities.leaky_rectify, 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                pos_dep_bias = False,
                padding = 1,
            )
            self.layers.append(layer)
        layer = HexConvLayer(
            incoming = self.layers[-1], 
            num_filters=num_filters, 
            radius = 2, 
            nonlinearity = lasagne.nonlinearities.leaky_rectify, 
            W=lasagne.init.HeNormal(gain='relu'), 
            b=lasagne.init.Constant(0),
            pos_dep_bias = False,
            padding = 0,
        )
        self.layers.append(layer)
        Qsigma_output_layer = HexConvLayer(
                incoming = self.layers[-1], 
                num_filters=1, 
                radius = 1, 
                nonlinearity = lambda x: 0.5*(lasagne.nonlinearities.sigmoid(x)-0.5), 
                W=lasagne.init.HeNormal(gain='relu'), 
                b=lasagne.init.Constant(0),
                pos_dep_bias = True,
                padding = 0,
            )
        self.layers.append(Qsigma_output_layer)
        Qsigma_output = lasagne.layers.get_output(Qsigma_output_layer)

        #If a loadfile is given, use saved parameter values
        if(loadfile is not None):
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

        #Build update function for both Pw and Qsigma
        Pw_loss = lasagne.objectives.aggregate(lasagne.objectives.binary_crossentropy(T.clip(Pw_output.flatten(2)[T.arange(Pw_targets.shape[0]),action_batch],0.0001,0.9999), T.clip(Pw_targets,0.0001,0.9999)), mode='mean')
        Pw_params = lasagne.layers.get_all_params(Pw_output_layer)

        Qsigma_loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(Qsigma_output.flatten(2)[T.arange(Qsigma_targets.shape[0]),action_batch], T.clip(Qsigma_targets,-0.25, 0.25)), mode='mean')
        Qsigma_params = lasagne.layers.get_all_params(Qsigma_output_layer)

        l2_penalty = regularize_layer_params(self.layers, l2)*1e-5

        loss = Pw_loss + Qsigma_loss + l2_penalty
        params = Pw_params + Qsigma_params
        if(loadfile is not None):
            updates, accu = rmsprop(loss, params, alpha, rho, epsilon, opt_vals.pop(0))
            self.opt_state.append(accu)
        else:
            updates, accu = rmsprop(loss, params, alpha, rho, epsilon)
            self.opt_state.append(accu)

        self._update = theano.function(
            [state_batch, action_batch, Pw_targets, Qsigma_targets],
            updates = updates,
            outputs = [Pw_loss, Qsigma_loss]
        )

        #Build mentor function for both Pw and Qsigma
        Pw_mentor_loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(Pw_output.flatten(),mentor_Pws.flatten()))
        Qsigma_mentor_loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(Qsigma_output.flatten(),mentor_Qsigmas.flatten()))

        loss = Pw_mentor_loss + Qsigma_mentor_loss
        params = Pw_params + Qsigma_params
        if(loadfile is not None):
            updates, accu = rmsprop(loss, params, alpha, rho, epsilon, opt_vals.pop(0))
            self.opt_state.append(accu)
        else:
            updates, accu = rmsprop(loss, params, alpha, rho, epsilon)
            self.opt_state.append(accu)

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

        Pw2, Qsigma2 = self._evaluate_multi(states2)
        Pw1 = self._evaluate_Pws(states1)
        #add a cap on the lowest possible value of losing probability
        Pl =  np.maximum(1-Pw2,0.00001)
        joint = np.prod(Pl, axis=1)

        #Update networks
        Pw_targets = np.zeros(terminals.size).astype(theano.config.floatX)
        Pw_targets[terminals==0] = joint[terminals==0]
        Pw_targets[terminals==1] = 1
        gamma = (joint[...,np.newaxis]/Pl)**2
        Qsigma_targets = np.zeros(terminals.size).astype(theano.config.floatX)
        Qsigma_targets[terminals==0] = joint[terminals==0]**2 - Pw1[np.arange(batch_size),actions][terminals==0]**2 + np.max(gamma*Qsigma2, axis=1)[terminals==0]
        Qsigma_targets[terminals==1] = 1 - Pw1[np.arange(batch_size),actions][terminals==1]**2
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

        #epsilon greedy
        if np.random.rand()<0.1:
            action = np.random.choice(np.where(played==0)[0])
            return action, Pw, Qsigma

        #add a cap on the lowest possible value of losing probability
        Pl =  np.maximum(1-Pw,0.00001)
        joint = np.prod(Pl)
        #if losing probability is sufficiently small just play the best move
        if joint < win_cutoff:
            values = np.copy(Pw)
            #never select played values
            values[played]=-2
            action = rargmax(values)
            return action, Pw, Qsigma

        gamma = (joint/Pl)**2
        values = gamma*Qsigma
        #never select played values
        values[played]=-2
        action = rargmax(values)
        return action, Pw, Qsigma

    def optimization_policy(self, state):
        played = np.logical_or(state[white,padding:-padding,padding:-padding], state[black,padding:-padding,padding:-padding]).flatten()
        state = np.asarray(state, dtype=theano.config.floatX)
        Pw = self._evaluate_Pw(state)
        values = Pw
        #never select played values
        values[played]=-2
        action = rargmax(values)
        return action

    def win_prob(self, state):
        state = np.asarray(state, dtype=theano.config.floatX)
        Pw = self._evaluate_Pw(state)
        values = Pw
        return values

    def win_prob_and_exp(self, state):
        state = np.asarray(state, dtype=theano.config.floatX)
        Pw, Qsigma = self._evaluate(state)
        Pw_values = Pw
        Qsigma_values = Qsigma
        return Pw_values, Qsigma_values


    def save(self, savefile = 'learner.save'):
        params = lasagne.layers.get_all_param_values(self.layers)
        data = {'params':params, 'mem':self.mem, 'opt': [[x.get_value() for x in y] for y in self.opt_state]}
        with open(savefile, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)