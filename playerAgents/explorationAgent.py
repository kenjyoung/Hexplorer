import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
from gamestate import gamestate
from copy import copy, deepcopy
import numpy as np
from inputFormat import *
from stateToInput import stateToInput
from learner import Learner

class explorationAgent:
	def __init__(self, state = gamestate(13)):
		self.state = copy(state)
		network = Learner(loadfile = os.path.dirname(os.path.realpath(__file__))+"/learner.save")
		self.evaluator = network.win_prob_and_exp

	def move(self, move):
		"""
		Make the passed move.
		"""
		self.state.play(move)

	def register(self, interface):
		interface.register_command("scores", self.gtp_scores)

	def gtp_scores(self, args):
		self.search()
		out_str = "gogui-gfx:\ndfpn\nVAR\nLABEL "
		for i in range(self.state.size*self.state.size):
			cell = np.unravel_index(i, (self.state.size,self.state.size))
			raw_cell = (cell[0]+(boardsize-self.state.size+1)//2, cell[1]+(boardsize-self.state.size+1)//2)
			toplay = white if self.state.toplay == self.state.PLAYERS["white"] else black
			if(toplay == black):
				cell = cell_m(cell)
			score_index = boardsize*raw_cell[0]+raw_cell[1]
			out_str+= chr(ord('a')+cell[0])+str(cell[1]+1)+" @"+str(self.scores[score_index])[0:6]+"@ "
		out_str+="\nTEXT scores\n"
		print(out_str)
		return(True, "")


	def search(self, time_budget = 1):
		"""
		Compute resistance for all moves in current state.
		"""
		state = stateToInput(self.state)
		#get equivalent white to play game if black to play
		toplay = white if self.state.toplay == self.state.PLAYERS["white"] else black
		if(toplay == black):
			state = mirror_game(state)
		played = np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
		state[black,padding:boardsize+padding,padding:boardsize+padding]).flatten()
		Pw, Qsigma = self.evaluator(state)
		Pl =  np.maximum(1-Pw,0.00001)
		joint = np.prod(Pl)
		gamma = (joint/Pl)**2
		print(Pl)
		print(joint)
		print(gamma)
		print(Qsigma)
		self.scores = gamma*Qsigma
		#set value of played cells impossibly low so they are never picked
		self.scores[played] = -2

	def best_move(self):
		"""
		Return the best move according to the current tree.
		"""
		move = np.unravel_index(self.scores.argmax(), (boardsize,boardsize))
		#correct move for smaller boardsizes
		move = (move[0]-(boardsize-self.state.size+1)//2, move[1]-(boardsize-self.state.size+1)//2)
		#flip returned move if black to play to get move in actual game
		toplay = white if self.state.toplay == self.state.PLAYERS["white"] else black
		if(toplay == black):
			move = cell_m(move)
		return move

	def set_gamestate(self, state):
		self.state = deepcopy(state)