import numpy as np
from inputFormat import *
import matplotlib.pyplot as plt
import cPickle
import argparse
import time
import os

def save(learner):
	print "saving network..."
	save_name = args.data+"/learner.save"
	learner.save(savefile = save_name)
	

def snapshot(learner):
	if not args.data:
		return
	print "saving network snapshot..."
	index = 0
	save_name = args.data+"/snapshot_"+str(index)+".save"
	while os.path.exists(save_name):
		index+=1
		save_name = args.data+"/snapshot_"+str(index)+".save"
	learner.save(savefile = save_name)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def show_plots():
	plt.figure(0)
	plt.plot(running_mean(costs,200))
	plt.ylabel('cost')
	plt.xlabel('episode')
	plt.draw()
	plt.pause(0.001)
	plt.figure(1)
	plt.plot(running_mean(values,200))
	plt.ylabel('value')
	plt.xlabel('episode')
	plt.draw()
	plt.pause(0.001)

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

print("loading starting positions... ")
datafile = open("data/scoredPositionsFull.npz", 'r')
data = np.load(datafile)
positions = data['positions']
datafile.close()
numPositions = len(positions)

if args.data and not os.path.exists(args.data):
	os.makedirs(args.data)

numEpisodes = 100000
batch_size = 64


#if load parameter is passed load a network from a file
if args.load:
	print("Loading agent...")
	Agent = Learner(loadfile = args.load)
else:
	print("Building agent...")
	Agent = Learner()

print("Running episodes...")
last_save = time.clock()
last_snapshot = time.clock()
show_plots()
try:
	for i in range(numEpisodes):
		num_step = 0
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
			action = Agent.exploration_policy(gameW if move_parity else gameB)
			state1 = np.copy(gameW if move_parity else gameB)
			move_cell = action_to_cell(action)
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
			Agent.update_memory(state1, action, state2, reward)
			Agent.learn(batch_size = batch_size)
			num_step += 1
			if(time.clock()-last_save > 60*save_time):
				save(Agent)
				last_save = time.clock()
			if(time.clock()-last_snapshot > 60*snapshot_time):
				snapshot(Agent)
				last_snapshot = time.clock()
		run_time = time.clock() - t
		print("Episode", i, "complete, Time per move: ", 0 if num_step == 0 else run_time/num_step)

except KeyboardInterrupt:
	#save snapshot of network if we interrupt so we can pickup again later
	save()
	exit(1)

save()
