import numpy as np
from learner import Learner
from inputFormat import *
import pickle
import argparse
import time
import os

def save(learner, Pw_vars, Qsigmas, Pw_costs, Qsigma_costs):
	print("saving network...")
	if(args.data):
		save_name = args.data+"/learner.save"
		data_name = args.data+"/data.save"
	else:
		save_name = "learner.save"
		data_name = "data.save"
	learner.save(savefile = save_name)
	data = {"Pw_vars":Pw_vars, "Qsigmas": Qsigmas, "Pw_costs":Pw_costs, "Qsigma_costs":Qsigma_costs}
	with open(data_name, 'wb') as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
	

def snapshot(learner):
	if not args.data:
		return
	print("saving network snapshot...")
	index = 0
	save_name = args.data+"/snapshot_"+str(index)+".save"
	while os.path.exists(save_name):
		index+=1
		save_name = args.data+"/snapshot_"+str(index)+".save"
	learner.save(savefile = save_name)

def action_to_cell(action):
	cell = np.unravel_index(action, (boardsize,boardsize))
	return(cell[0]+padding, cell[1]+padding)

def flip_action(action):
	return boardsize*boardsize-1-action

parser = argparse.ArgumentParser()
parser.add_argument("--load", "-l", type=str, help="Specify a file with a prebuilt network to load.")
parser.add_argument("--data", "-d", type =str, help="Specify a directory to save/load data for this run.")
args = parser.parse_args()

#save network every x minutes during training
save_time = 60
#save snapshot of network to unique file every x minutes during training
snapshot_time = 240

print("Loading starting positions... ")
datafile = open("data/scoredPositionsFull.npz", 'rb')
data = np.load(datafile)
positions = data['positions']
datafile.close()
numPositions = len(positions)

if args.data and not os.path.exists(args.data):
	os.makedirs(args.data)

if args.data and os.path.exists(args.data+'/data.save'):
	with open(args.data+'/data.save', 'rb') as f:
		data = pickle.load(f)
		Pw_costs = data['Pw_costs']
		Qsigma_costs = data['Qsigma_costs']
		Qsigmas = data['Qsigmas']
		Pw_vars = data['Pw_vars']
else:
	Pw_costs = []
	Qsigma_costs = []
	Qsigmas = []
	Pw_vars = []

numEpisodes = 100000
batch_size = 64
boardsize = 13


#if load parameter is passed or a saved learner is available in the data directory load a network from a file
if args.load:
	print("Loading agent...")
	Agent = Learner(loadfile = args.load)
elif args.data and os.path.exists(args.data+'/learner.save'):
	print("Loading agent...")
	Agent = Learner(loadfile = args.data+'/learner.save')
else:
	print("Building agent...")
	Agent = Learner()

print("Running episodes...")
last_save = time.clock()
last_snapshot = time.clock()
try:
	for i in range(numEpisodes):
		num_step = 0
		Pw_cost_sum = 0
		Qsigma_cost_sum = 0
		Qsigma_sum = 0
		Pw_var_sum = 0
		#randomly choose who is to move from each position to increase variability in dataset
		move_parity = np.random.choice([True,False])
		#randomly choose starting position from database
		index = np.random.randint(numPositions)
		#randomly flip states to capture symmetry
		if(np.random.choice([True,False])):
			gameW = np.copy(positions[index])
		else:
			gameW = flip_game(positions[index])
		gameB = mirror_game(gameW)
		t = time.clock()
		while(winner(gameW)==None):
			action, Pw, Qsigma = Agent.exploration_policy(gameW if move_parity else gameB)
			state1 = np.copy(gameW if move_parity else gameB)
			move_cell = action_to_cell(action)
			print(action)
			print(state_string(gameW, boardsize))
			play_cell(gameW, move_cell if move_parity else cell_m(move_cell), white if move_parity else black)
			play_cell(gameB, cell_m(move_cell) if move_parity else move_cell, black if move_parity else white)
			if(not winner(gameW)==None):
				terminal = 1
			else:
				terminal = 0
			#randomly flip states to capture symmetry
			if(np.random.choice([True,False])):
				state2 = np.copy(gameB if move_parity else gameW)
			else:
				state2 = flip_game(gameB if move_parity else gameW)
			move_parity = not move_parity
			Agent.update_memory(state1, action, state2, terminal)
			costs = Agent.learn(batch_size = batch_size)
			if(costs is not None):
				Pw_cost, Qsigma_cost = costs
			else:
				Pw_cost = 0
				Qsigma_cost = 0

			#update running sums for this episode
			num_step += 1
			Pw_var_sum += np.mean((1-Pw)*Pw)
			Qsigma_sum += np.mean(Qsigma)
			Qsigma_cost_sum += Qsigma_cost
			Pw_cost_sum += Pw_cost

			if(time.clock()-last_save > 60*save_time):
				save(Agent, Pw_vars, Qsigmas, Pw_costs, Qsigma_costs)
				last_save = time.clock()
			if(time.clock()-last_snapshot > 60*snapshot_time):
				snapshot(Agent)
				last_snapshot = time.clock()
		run_time = time.clock() - t
		print("Episode"+str(i)+"complete, Time per move: "+str(0 if num_step == 0 else run_time/num_step)+" Pw Cost: "+str(Pw_cost)+" Qsigma Cost: "+str(Qsigma_cost))

		#log data for this episode
		if(num_step!=0):
			Pw_vars.append(Pw_var_sum/num_step)
			Qsigmas.append(Qsigma_sum/num_step)
			Qsigma_costs.append(Qsigma_cost_sum/num_step)
			Pw_costs.append(Pw_cost_sum/num_step)

except KeyboardInterrupt:
	#save snapshot of network if we interrupt so we can pickup again later
	save(Agent, Pw_vars, Qsigmas, Pw_costs, Qsigma_costs)
	exit(1)

save(Agent, Pw_vars, Qsigmas, Pw_costs, Qsigma_costs)
