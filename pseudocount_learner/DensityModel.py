import math
import numpy as np
import cts.model as model
from inputFormat import *

"""
This model is largely based on one by Marc G. Bellemare,
available at: https://github.com/mgbellemare/SkipCTS/blob/master/python/tutorials/density_modelling_tutorial.ipynb
"""

#TODO: speed up with some preprocessing of state

def upper_left_context(state, x, y):
    n =((-1,0), (0,-1), (-1,1))
    context = [0,0,0]
    for i in range(3):
        color = 0 if state[white, x+padding+n[i][0], y+padding+n[i][1]] else 1 if state[black, x+padding+n[i][0], y+padding+n[i][1]] else 2
        context[i] = color
    return context

def full_context(state):
    context=[0 for x in range(boardsize*boardsize)]
    for x in range(boardsize):
        for y in range(boardsize):
            color = 1 if state[white, x+padding, y+padding] else 2 if state[black, x+padding, y+padding] else 0
            context[x+boardsize*y] = color
    return context

def color(state, x, y):
    return 1 if state[white, x+padding+n[i][0], y+padding+n[i][1]] else 2 if state[black, x+padding+n[i][0], y+padding+n[i][1]] else 0


class DensityModel:
    """A density model for Freeway frames.
    
    This is exactly the same as the ConvolutionalDensityModel, except that we use one model for each
    pixel location.
    """
    def __init__(self, context_functor=upper_left_context, state_alphabet=None, action_alphabet=None):
        """Constructor.
        
        Args:
            init_frame: A sample frame (numpy array) from which we determine the shape and type of our data.
            context_functor: Function mapping image x position to a context.
        """

        context_length = len(context_functor(new_game(), -1, -1))
        self.state_models = np.zeros((boardsize, boardsize), dtype=object)
        
        for y in range(frame_shape[0]):
            for x in range(frame_shape[1]):
                self.models[x, y] = model.CTS(context_length=context_length, alphabet=state_alphabet)

        self.action_model = model.CTS(context_length=context_length, alphabet=action_alphabet)
        
        self.context_functor = context_functor

    def update(self, state, action):
        total_log_probability = 0.0
        for y in range(boardsize):
            for x in range(boardsize):
                context = self.context_functor(state, x, y)
                value = color(state, x, y) 
                total_log_probability += self.state_models[x, y].update(context=context, symbol=value)
        total_log_probability += self.action_model.update(context=full_context(state), symbol=action)

        return total_log_probability

    def action_pseudocounts(self, state):
        total_log_probability = 0.0
        for y in range(boardsize):
            for x in range(boardsize):
                context = self.context_functor(state, x, y)
                value = color(state, x, y) 
                total_log_probability += self.state_models[x, y].update(context=context, symbol=value)
        context = full_context(state)
        recording_probs = [math.exp(total_log_probability + self.action_model.recording_log_prob(context=context, symbol=x)) for x in range(boardsize*boardsize)]
        probs = [math.exp(total_log_probability + self.action_model.log_prob(context=context, symbol=x)) for x in range(boardsize*boardsize)]
        pseudocounts = [p*(1-p_r)/(p_r-p) for (p,p_r) in zip(probs, recording_probs)]

    def sample(self):
        state = new_game()
        for x in range(boardsize):
            for y in range(boardsize):
                context = self.context_functor(state, x, y)
                value = self.state_models[x, y].sample(context=context, rejection_sampling=True)
                if value==white:
                    play_cell(state, (x,y), white)
                elif value==black:
                    play_cell(state, (x,y), black)
        action = self.action_model.sample(full_context(state), rejection_sampling=True)
        return state, action