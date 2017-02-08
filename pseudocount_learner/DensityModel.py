import math
import numpy as np
import cts.model as model
from inputFormat import *
import time

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
    return 1 if state[white, x+padding, y+padding] else 2 if state[black, x+padding, y+padding] else 0


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
        
        for y in range(boardsize):
            for x in range(boardsize):
                self.state_models[x, y] = model.CTS(context_length=context_length, alphabet=set(state_alphabet))

        self.action_model = model.CTS(context_length=boardsize*boardsize, alphabet=set(action_alphabet))
        
        self.context_functor = context_functor

    def update(self, state):
        total_log_probability = 0.0
        total_log_recording_probability = 0.0
        for y in range(boardsize):
            for x in range(boardsize):
                context = self.context_functor(state, x, y)
                value = color(state, x, y)
                total_log_probability += self.state_models[x, y].update(context=context, symbol=value)
                total_log_recording_probability += self.state_models[x, y].log_prob(context=context, symbol=value)
        p = np.exp(total_log_probability)
        p_r = np.exp(total_log_recording_probability)
        pseudocount = p*(1-p_r)/(p_r-p)
        return pseudocount

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