import theano
import time
import numpy as np
from inputFormat import *
from learner import Learner
import pickle
import argparse
import os

def save(learner, Pw_costs, Qsigma_costs):
	print("saving network...")
	if(args.data):
		save_name = args.data+"/learner.save"
		data_name = args.data+"/data.save"
	else:
		save_name = "learner.save"
		data_name = "data.save"
	learner.save(savefile = save_name)
	data = {"Pw_costs":Pw_costs, "Qsigma_costs":Qsigma_costs}
	with open(data_name, 'wb') as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


parser = argparse.ArgumentParser()
parser.add_argument("--load", "-l", type=str, help="Specify a file with a prebuilt network to load.")
parser.add_argument("--data", "-d", type =str, help="Specify a directory to save/load data for this run.")
args = parser.parse_args()

print("Loading data... ")
datafile = open("data/scoredPositionsFull.npz", 'rb')
data = np.load(datafile)
positions = data['positions']
scores = data['scores']
datafile.close()

if args.data and not os.path.exists(args.data):
	os.makedirs(args.data)

if args.data and os.path.exists(args.data+'/data.save'):
	with open(args.data+'/data.save', 'rb') as f:
		data = pickle.load(f)
		Pw_costs = data['Pw_costs']
		Qsigma_costs = data['Qsigma_costs']
else:
	Pw_costs = []
	Qsigma_costs = []


positions = positions.astype(theano.config.floatX)
scores = scores.astype(theano.config.floatX)
n_train = scores.shape[0]

numEpochs = 10
iteration = 0
batch_size = 64
numBatches = n_train//batch_size

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

print("Training model on mentor set...")
indices = list(range(n_train))
try:
	for epoch in range(numEpochs):
		Pw_cost_sum = 0
		Qsigma_cost_sum = 0
		print("epoch: "+str(epoch))
		np.random.shuffle(indices)
		for batch in range(numBatches):
			t = time.clock()
			state_batch = positions[indices[batch*batch_size:(batch+1)*batch_size]]
			Pw_batch = scores[indices[batch*batch_size:(batch+1)*batch_size]]
			Qsigma_batch = np.ones(Pw_batch.shape)
			Pw_cost, Qsigma_cost = Agent.mentor(state_batch, Pw_batch, Qsigma_batch)
			Pw_cost_sum += Pw_cost
			Qsigma_cost_sum += Qsigma_cost
			run_time = time.clock()-t
			iteration+=1
			print("Time per position: "+str(run_time/batch_size)+" Pw Cost: "+str(Pw_cost)+" Qsigma Cost: "+str(Qsigma_cost))
		Pw_costs.append(Pw_cost_sum/numBatches)
		Qsigma_costs.append(Qsigma_cost_sum/numBatches)

		#save snapshot of network every epoch in case something goes wrong
		save(Agent, Pw_costs, Qsigma_costs)
except KeyboardInterrupt:
	#save snapshot of network if we interrupt so we can pickup again later
	save(Agent, Pw_costs, Qsigma_costs)
	exit(1)

print("done training!")
save(Agent, Pw_costs, Qsigma_costs)