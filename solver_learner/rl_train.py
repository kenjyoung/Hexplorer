import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
import numpy as np
from Q_learner import Learner
from inputFormat import *
import pickle
import argparse
import time
import os
import subprocess
from program import Program
import threading
import shutil


class solver:
    def __init__(self, exe):
        self.exe = exe 
        self.program = Program(self.exe, False)
        self.lock  = threading.Lock()
        self.thread = None

    class solveThread(threading.Thread):
        def __init__(self, solver, color):
            threading.Thread.__init__(self)
            self.color = color
            self.solver = solver
        def run(self):
            try:
                self.moves = self.solver.sendCommand("dfpn-solver-find-winning " + ("black" if self.color == black else "white")).split()
            except Program.Died:
                pass

    def start_solve(self, color):
        self.thread = self.solveThread(self, color)
        self.thread.start()

    def stop_solve(self):
        self.interrupt()
        self.thread.join()
        return self.thread.moves

    def set_game(self, game):
        self.sendCommand("clear_board")
        for i in range(boardsize):
            for j in range(boardsize):
                cell = (i+padding,j+padding)
                move_str = move(cell)
                color = check_cell(game, cell)
                if color is white:
                    self.sendCommand("play white "+move_str)
                elif color is black:
                    self.sendCommand("play black "+move_str)

    def play_move(self, move, color):
        if color is white:
            self.sendCommand("play white "+move)
        elif color is black:
            self.sendCommand("play black "+move)

    def sendCommand(self, command):
        self.lock.acquire()
        answer = self.program.sendCommand(command)
        self.lock.release()
        return answer

    def interrupt(self):
        self.program.interrupt()

    def reconnect(self):
        self.program.terminate()
        self.program = Program(self.exe,True)
        self.lock = threading.Lock()


def get_git_hash():
    return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())


def save(learner, solver, Pw_vars, Pw_costs):
    print("saving network...")
    if(args.data):
        save_name = args.data+"/learner.save"
        data_name = args.data+"/data.save"
        solver_name = args.data+"/solver.save"
    else:
        save_name = "learner.save"
        data_name = "data.save"
        solver_name = "solver.save"
    learner.save(savefile = save_name)
    data = {"Pw_vars":Pw_vars, "Pw_costs":Pw_costs}
    with open(data_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    solver.sendCommand('dfpn-dump-tt')
    shutil.move("tt.dump",solver_name)



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
snapshot_interval = 500

# print("Loading starting positions... ")
# datafile = open("../data/scoredPositionsFull.npz", 'rb')
# data = np.load(datafile)
# positions = data['positions']
# datafile.close()
# numPositions = len(positions)

wolve_exe = "/home/kenny/Hex/benzene-vanilla/src/wolve/wolve 2>/dev/null" 
solver = solver(wolve_exe)
solver.sendCommand("boardsize "+str(boardsize))

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
        Pw_vars = data['Pw_vars']

if args.data and os.path.exists(args.data+'/solver.save'):
    shutil.copy(args.data+'/solver.save', "tt.dump")
    solver.sendCommand("dfpn-restore-tt")

else:
    Pw_costs = []
    Pw_vars = []

numEpisodes = 100000
batch_size = 32
boardsize = 7


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
        Pw_var_sum = 0
        # #randomly choose who is to move from each position to increase variability in dataset
        # move_parity = np.random.choice([True,False])
        # #randomly choose starting position from database
        # index = np.random.randint(numPositions)
        # #randomly flip states to capture symmetry
        # if(np.random.choice([True,False])):
        #     gameW = np.copy(positions[index])
        # else:
        #     gameW = flip_game(positions[index])
        gameW = new_game(7)
        wins = []
        play_cell(gameW, action_to_cell(np.random.randint(0,boardsize*boardsize)), white)
        solver.set_game(gameW)
        move_parity = False
        gameB = mirror_game(gameW)
        t = time.clock()
        next_terminal = 0
        while(winner(gameW)==None):
            action, Pw = Agent.exploration_policy(gameW if move_parity else gameB, move_set = wins)
            move_cell = action_to_cell(action)
            state1 = np.copy(gameW if move_parity else gameB)
            played = np.logical_or(state1[white,padding:-padding,padding:-padding], state1[black,padding:-padding,padding:-padding]).flatten()
            solver.play_move(move(move_cell) if move_parity else move(cell_m(move_cell)), white if move_parity else black)
            solver.start_solve(black if move_parity else white)
            play_cell(gameW, move_cell if move_parity else cell_m(move_cell), white if move_parity else black)
            play_cell(gameB, cell_m(move_cell) if move_parity else move_cell, black if move_parity else white)
            move_parity = not move_parity
            if(not winner(gameW)==None or next_terminal == 1):
                terminal = 1
                next_terminal = 0
            else:
                terminal = 0
            if(len(wins)>0):
                next_terminal = 1
            #randomly flip states to capture symmetry
            if(np.random.choice([True,False])):
                state2 = np.copy(gameB if move_parity else gameW)
            else:
                state2 = flip_game(gameB if move_parity else gameW)
            Agent.update_memory(state1, action, state2, terminal)
            Pw_cost = Agent.learn(batch_size = batch_size)
            if(Pw_cost is None):
                Pw_cost = 0

            #update running sums for this episode
            num_step += 1
            Pw_var_sum += np.mean(((1-Pw)*Pw)[np.logical_not(played)])
            Pw_cost_sum += Pw_cost

            if(time.clock()-last_save > 60*save_time):
                save(Agent, solver, Pw_vars, Pw_costs)
                last_save = time.clock()
            wins = [unpadded_cell(x) for x in solver.stop_solve()]
            if not move_parity:
                wins = [cell_m(x) for x in wins]
        if(i%snapshot_interval == 0):
            snapshot(Agent)
            save(Agent, solver, Pw_vars, Pw_costs)
        run_time = time.clock() - t
        print("Episode"+str(i)+"complete, Time per move: "+str(0 if num_step == 0 else run_time/num_step)+" Pw Cost: "+str(Pw_cost))

        #log data for this episode
        if(num_step!=0):
            Pw_vars.append(Pw_var_sum/num_step)
            Pw_costs.append(Pw_cost_sum/num_step)

except KeyboardInterrupt:
    #save snapshot of network if we interrupt so we can pickup again later
    save(Agent, solver, Pw_vars, Pw_costs)
    exit(1)

save(Agent, solver, Pw_vars, Pw_costs)
