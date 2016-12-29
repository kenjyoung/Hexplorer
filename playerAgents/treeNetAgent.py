import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
from gamestate import gamestate
from copy import copy, deepcopy
import numpy as np
from inputFormat import *
from stateToInput import stateToInput
import time

class node:
    """
    Node for the tree search. Stores the move applied to reach this node from its parent,
    stats for the associated game position, children, parent and outcome 
    (outcome==none unless the position ends the game).
    """
    def __init__(self, Pw = None, Qsigma = None,  move = None, parent = None):
    	"""
    	Initialize a new node with optional move and parent and initially empty
    	children list and rollout statistics and unspecified outcome.
    	"""
        self.move = move
        self.parent = parent
        self.children = []
        self.outcome = gamestate.PLAYERS["none"]
        self.Pw = Pw
        self.Qsigma = Qsigma

    def add_children(self, children):
    	"""
    	Add a list of nodes to the children of this node.
    	"""
    	self.children += children

    def set_outcome(self, outcome):
    	"""
    	Set the outcome of this node (i.e. if we decide the node is the end of
    	the game)
    	"""
    	self.outcome = outcome

    def exp_values(self):
        child_Pws = np.asarray([child.Pw for child in self.children])
        child_Qsigmas = np.asarray([child.Qsigma for child in self.children])
        joint = np.prod(child_Pws)
        gamma = (joint/child_Pws)**2
        values = gamma*child_Qsigmas
        return values

    def win_probs(self):
        child_Pws = np.asarray([child.Pw for child in self.children])
        return child_Pws

class treeNetAgent:
    def __init__(self, state = gamestate(13)):
    	self.state = copy(state)
    	self.root = node()
        network = Learner(loadfile = os.path.realpath(__file__)+"/learner.save")
        self.evaluator = network.win_prob_and_exp

    def move(self, move):
    	"""
    	Make the passed move.
    	"""
    	for child in self.root.children:
    		#make the child associated with the move the new root
    		if move == child.move:
    			child.parent = None
    			self.root = child
    			self.state.play(child.move)
    			return

    	#if for whatever reason the move is not in the children of
    	#the root just throw out the tree and start over
    	self.state.play(move)
    	self.root = node()

    def register(self, interface):
    	interface.register_command("scores", self.gtp_scores)

    def gtp_scores(self, args):
    	self.search(10)
    	out_str = "gogui-gfx:\ndfpn\nVAR\nLABEL "
    	for node in self.root.children:
    		cell = np.unravel_index(node.move, (boardsize,boardsize))
    		out_str+= chr(ord('a')+cell[0])+str(cell[1]+1)+" @"+str(node.value(0))[0:6]+"@ "
    	out_str+="\nTEXT scores\n"
    	print(out_str)
    	return(True, "")

    def evaluate(self, parent, state):
        #if anybody wins, this node must be a win for the player who just played
        if state.winner() != state.PLAYERS["none"]:
            return 1.0, 0.0
        children = []
        #get equivalent white to play game if black to play
        toplay = white if state.toplay == self.state.PLAYERS["white"] else black
        state = stateToInput(state)
        if(toplay == black):
        	state = mirror_game(state)
        played = np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
        state[black,padding:boardsize+padding,padding:boardsize+padding]).flatten()
        Pws, Qsigmas = self.evaluator(state)
        Pw = np.prod(1-Pws)
        gamma = (joint/(1-Pw))**2
        #don't include the gain from the current move since
        #we are concerned with what we can gain from further search
        Qsigmas = np.max((gamma*Qsigmas))
        for i in range(len(scores)):
        	if not played[i]:
        		if(toplay == white):
        			move = i
        		else:
        			move = boardsize*(i%boardsize)+i//boardsize
        		children.append(node(Pw = Pw[i], Qsigma = Qsigma[i], move = move, parent = parent))
        parent.add_children(children)
        return Pw, Qsigma


    def search(self, time_budget = 1):
        #TODO: complete backup phase and debug and/or rework everything else
    	startTime = time.time()
    	num_evals = 0

    	#do until we exceed our time budget
    	while(time.time() - startTime < time_budget):
    		node, state = self.select_node()
    		Pw, Qsigma = self.evaluate(node, state)
    		self.backup(node, value)
    		num_evals += 1
    	sys.stderr.write("Ran "+str(num_evals)+ " evaluations in " +\
    		str(time.time() - startTime)+" sec\n")

    def select_node(self):
    	"""
    	Select a node in the tree to preform an evaluation from.
    	"""
    	node = self.root
    	state = deepcopy(self.state)

    	#stop if we reach a leaf node
    	while(len(node.children)!=0):
    		#decend to the maximum value node
            exp_values = node.exp_values()
            node = node.children[np.argmax(exp_values)]
            move = np.unravel_index(node.move, (boardsize, boardsize))
    		#correct move for smaller boardsizes
            move = (move[0]-(boardsize-self.state.size+1)/2, move[1]-(boardsize-self.state.size+1)/2)
            state.play(move)
    	return (node, state)

    def backup(self, node, value):
    	"""
    	Update the node statistics on the path from the passed node to root to reflect
    	the outcome of a network evaluation.
    	"""
    	while node!=None:
    		node.N += 1
    		node.Q += value
    		value = -value
    		node = node.parent

    def best_move(self):
    	"""
    	Return the best move according to the current tree.
    	"""
    	if(self.state.winner() != gamestate.PLAYERS["none"]):
    		return gamestate.GAMEOVER

    	#choose the move of the highest value node
    	move = np.unravel_index(max(self.root.children, key = lambda n: n.value(0)).move, (boardsize, boardsize))
    	#correct move for smaller boardsizes
    	move = (move[0]-(boardsize-self.state.size+1)/2, move[1]-(boardsize-self.state.size+1)/2)
    	return move

    def set_gamestate(self, state):
    	self.state = deepcopy(state)
    	self.root = node()