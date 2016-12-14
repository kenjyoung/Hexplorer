import theano
import time
import numpy as np
from inputFormat import *
from learner import Learner
import pickle
import argparse
import os

def save(learner):
	print("saving network...")
	save_name = args.data+"/learner.save"
	learner.save(savefile = save_name)


parser = argparse.ArgumentParser()
parser.add_argument("--load", "-l", type=str, help="Specify a file with a prebuilt network to load.")
parser.add_argument("--data", "-d", type =str, help="Specify a directory to save/load data for this run.")
args = parser.parse_args()

print("loading data... ")
datafile = open("data/scoredPositionsFull.npz", 'rb')
data = np.load(datafile)
positions = data['positions']
scores = data['scores']
datafile.close()

if args.data and not os.path.exists(args.data):
	os.makedirs(args.data)

positions = positions.astype(theano.config.floatX)
scores = scores.astype(theano.config.floatX)
n_train = scores.shape[0]

numEpochs = 100
iteration = 0
batch_size = 64
numBatches = n_train//batch_size

#if load parameter is passed load a network from a file
if args.load:
	print("Loading agent...")
	Agent = Learner(loadfile = args.load)
else:
	print("Building agent...")
	Agent = Learner()

print("Training model on mentor set...")
indices = list(range(n_train))
try:
	for epoch in range(numEpochs):
		print("epoch: ",epoch)
		np.random.shuffle(indices)
		for batch in range(numBatches):
			t = time.clock()
			state_batch = positions[indices[batch*batch_size:(batch+1)*batch_size]]
			Pw_batch = scores[indices[batch*batch_size:(batch+1)*batch_size]]
			Qsigma_batch = np.ones(Pw_batch.shape)
			Agent.mentor(state_batch, Pw_batch, Qsigma_batch)
			run_time = time.clock()-t
			iteration+=1
			print("Time per position: ", run_time/(batch_size))
		#save snapshot of network every epoch in case something goes wrong
		save(Agent)
except KeyboardInterrupt:
	#save snapshot of network if we interrupt so we can pickup again later
	save(Agent)
	exit(1)

print("done training!")
save(Agent)