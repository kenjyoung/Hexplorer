import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
import numpy as np
from pseudocount_learner import Learner
from inputFormat import *
import pickle
import argparse
import time
import os
import subprocess

def get_git_hash():
    return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())

def save(learner, Pw_vars, Counts, Pw_costs, Count_costs):
    print("saving network...")
    if(args.data):
        save_name = args.data+"/learner.save"
        data_name = args.data+"/data.save"
    else:
        save_name = "learner.save"
        data_name = "data.save"
    learner.save(savefile = save_name)
    data = {"Pw_vars":Pw_vars, "Counts": Counts, "Pw_costs":Pw_costs, "Count_costs":Count_costs}
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
snapshot_interval = 1000

# print("Loading starting positions... ")
# datafile = open("data/scoredPositionsFull.npz", 'rb')
# data = np.load(datafile)
# positions = data['positions']
# datafile.close()
# numPositions = len(positions)

if args.data and not os.path.exists(args.data):
    os.makedirs(args.data)
    with open(args.data+'/info.txt', 'a') as f:
        f.write('git commit: '+get_git_hash()+'\n')
        f.write('load: '+('None' if args.load is None else args.load+'\n'))
        f.flush()

if args.data and os.path.exists(args.data+'/data.save'):
    with open(args.data+'/data.save', 'rb') as f:
        data = pickle.load(f)
        Pw_costs = data['Pw_costs']
        Count_costs = data['Count_costs']
        Counts = data['Counts']
        Pw_vars = data['Pw_vars']
else:
    Pw_costs = []
    Count_costs = []
    Counts = []
    Pw_vars = []

numEpisodes = 1000000
batch_size = 32
boardsize = 5


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
try:
    for i in range(len(Pw_costs), numEpisodes):
        num_step = 0
        Pw_cost_sum = 0
        Count_cost_sum = 0
        Count_sum = 0
        Pw_var_sum = 0
        # #randomly choose who is to move from each position to increase variability in dataset
        # move_parity = np.random.choice([True,False])
        #randomly choose starting position from database
        # index = np.random.randint(numPositions)
        # #randomly flip states to capture symmetry
        # if(np.random.choice([True,False])):
        #     gameW = np.copy(positions[index])
        # else:
        #     gameW = flip_game(positions[index])
        gameW = new_game(5)
        play_cell(gameW, action_to_cell(np.random.randint(0,25)), white)
        gameB = mirror_game(gameW)
        move_parity = False
        t = time.clock()
        while(winner(gameW)==None):
            action, Pw, Count = Agent.exploration_policy(gameW if move_parity else gameB)
            state1 = np.copy(gameW if move_parity else gameB)
            played = np.logical_or(state1[white,padding:-padding,padding:-padding], state1[black,padding:-padding,padding:-padding]).flatten()
            move_cell = action_to_cell(action)
            #print(state_string(gameW, boardsize))
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
                Pw_cost, Count_cost = costs
            else:
                Pw_cost = 0
                Count_cost = 0

            #update running sums for this episode
            num_step += 1
            Pw_var_sum += np.mean(((1-Pw)*Pw)[np.logical_not(played)])
            Count_sum += np.mean(Count[np.logical_not(played)])
            Count_cost_sum += Count_cost
            Pw_cost_sum += Pw_cost

            if(time.clock()-last_save > 60*save_time):
                save(Agent, Pw_vars, Counts, Pw_costs, Count_costs)
                last_save = time.clock()
        if(i%snapshot_interval == 0):
            snapshot(Agent)
            save(Agent, Pw_vars, Counts, Pw_costs, Count_costs)
        run_time = time.clock() - t
        print("Episode"+str(i)+"complete, Time per move: "+str(0 if num_step == 0 else run_time/num_step)+" Pw Cost: "+str(0 if num_step == 0 else Pw_cost_sum/num_step)+" Count Cost: "+str(0 if num_step == 0 else Count_cost_sum/num_step))

        #log data for this episode
        if(num_step!=0):
            Pw_vars.append(Pw_var_sum/num_step)
            Counts.append(Count_sum/num_step)
            Count_costs.append(Count_cost_sum/num_step)
            Pw_costs.append(Pw_cost_sum/num_step)

except KeyboardInterrupt:
    #save snapshot of network if we interrupt so we can pickup again later
    save(Agent, Pw_vars, Counts, Pw_costs, Count_costs)
    exit(1)

save(Agent, Pw_vars, Counts, Pw_costs, Count_costs)
