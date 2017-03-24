import argparse
from program import Program
import threading
import pickle
from gamestate import gamestate
import sys

class agent:
    def __init__(self, exe):
        self.exe = exe 
        self.program = Program(self.exe, True)
        self.name = self.program.sendCommand("name").strip()
        self.lock  = threading.Lock()

    def sendCommand(self, command):
        self.lock.acquire()
        answer = self.program.sendCommand(command)
        self.lock.release()
        return answer

    def reconnect(self):
        self.program.terminate()
        self.program = Program(self.exe,True)
        self.lock = threading.Lock()

def move_to_cell(move):
    x = ord(move[0].lower())-ord('a')
    y = int(move[1:])-1
    return (x,y)

def run_game(blackAgent, whiteAgent, boardsize, opening=None, verbose = False):
    game = gamestate(boardsize)
    winner = None
    moves = []
    blackAgent.sendCommand("clear_board")
    whiteAgent.sendCommand("clear_board")
    if(opening is not None):
        game.place_white(move_to_cell(opening))
        whiteAgent.sendCommand("play white "+opening)
        blackAgent.sendCommand("play white "+opening)
    while(True):
        move = blackAgent.sendCommand("genmove black").strip()
        if( move == "resign"):
            winner = game.PLAYERS["white"]
            return winner
        moves.append(move)
        game.place_black(move_to_cell(move))
        whiteAgent.sendCommand("play black "+move)
        if verbose:
            print(blackAgent.name+" v.s. "+whiteAgent.name)
            print(game)
        if(game.winner() != game.PLAYERS["none"]):
            winner = game.winner()
            break
        sys.stdout.flush()
        move = whiteAgent.sendCommand("genmove white").strip()
        if( move == "resign"):
            winner = game.PLAYERS["black"] 
            return winner
        moves.append(move)
        game.place_white(move_to_cell(move))
        blackAgent.sendCommand("play white "+move)
        if verbose:
            print(blackAgent.name+" v.s. "+whiteAgent.name)
            print(game)
        if(game.winner() != game.PLAYERS["none"]):
            winner = game.winner()
            break
        sys.stdout.flush()
    winner_name = blackAgent.name if winner == game.PLAYERS["black"] else whiteAgent.name
    loser_name =  whiteAgent.name if winner == game.PLAYERS["black"] else blackAgent.name
    print("Game over, " + winner_name+ " ("+game.PLAYER_STR[winner]+") " + "wins against "+loser_name)
    print(game)
    print(" ".join(moves))
    return winner

wolve_exe = "/home/kenny/Hex/benzene-vanilla/src/wolve/wolve 2>/dev/null"
hexplorer_exe = "/home/kenny/Hex/Hexplorer/playerAgents/program.py 2>/dev/null"
Q_learner_dir = "/home/kenny/Hex/Hexplorer/Q_learner/rl_train7x7_1/"
count_learner_dir = "/home/kenny/Hex/Hexplorer/pseudocount_learner/rl_train7x7_1/"

parser = argparse.ArgumentParser(description="Run tournament against wolve at various learning stages and output results.")
parser.add_argument("--time", "-t", type=int, help="total time allowed for wolve each move in seconds.")
parser.add_argument("--load", "-l", type=str, help="Specify a file with a partially completed run to load.")
parser.add_argument("--verbose", "-v", dest="verbose", action='store_const',
                    const=True, default=False,
                    help="print board after each move.")
parser.add_argument

args = parser.parse_args()

temperature = 300
snapshot_interval = 500
num_snapshots = 200

print("Starting test...")

#Initialize wolve
wolve = agent(wolve_exe)
if(args.time):
    move_time = args.time
else:
    move_time = 1
wolve.sendCommand("param_wolve max_time "+str(move_time))
wolve.sendCommand("boardsize 7")

#Initialize count based player
counthex = agent(hexplorer_exe)
counthex.sendCommand("boardsize 7")

#Initialize ordinary Q-learning based player
Qhex = agent(hexplorer_exe)
Qhex.sendCommand("boardsize 7")

#Create list of all opening moves
moves=[chr(i+ord('a'))+str(j+1) for i in range(7) for j in range(7)]
#moves = ['a1']

if(args.load):
    with open(args.load, 'rb') as f:
        data = pickle.load(f)
        white_win_rates_Q = data['white_win_rates_Q']
        black_win_rates_Q = data['black_win_rates_Q']
        white_win_rates_count = data['white_win_rates_count']
        black_win_rates_count = data['black_win_rates_count']
        episodes = data['episodes']
else:
    white_win_rates_Q = []
    black_win_rates_Q = []
    white_win_rates_count = []
    black_win_rates_count = []
    episodes = []

for i in range(len(episodes), num_snapshots):
    snapshot_num = i
    episodes.append(snapshot_num*snapshot_interval)
    Qloadfile = Q_learner_dir+'snapshot_'+str(snapshot_num)+'.save'
    countloadfile = count_learner_dir+'snapshot_'+str(snapshot_num)+'.save'
    counthex.sendCommand('agent count '+countloadfile)
    Qhex.sendCommand('agent Q '+Qloadfile)
    #Initialize win counts
    white_wins_Q = 0
    black_wins_Q = 0
    white_wins_count = 0
    black_wins_count = 0
    #Run games for count based agent
    for move in moves:
        wolve.reconnect()
        wolve.sendCommand("param_wolve max_time "+str(move_time))
        wolve.sendCommand("param_wolve temperature "+str(temperature))
        wolve.sendCommand("boardsize 7")
        winner = run_game(wolve, counthex, 7, move, args.verbose)
        if(winner == gamestate.PLAYERS["white"]):
            white_wins_count += 1
        winner = run_game(counthex, wolve, 7, move, args.verbose)
        if(winner == gamestate.PLAYERS["black"]):
            black_wins_count += 1
    white_win_rates_count.append(white_wins_count/(7*7))
    black_win_rates_count.append(black_wins_count/(7*7))
    print("agent count, opening "+move+", black win rate: "+str(black_wins_count/(7*7))+", white win rate: "+str(white_wins_count/(7*7)))


    #Run games for Q-learning based agent
    for move in moves:
        wolve.reconnect()
        wolve.sendCommand("param_wolve max_time "+str(move_time))
        wolve.sendCommand("param_wolve temperature "+str(temperature))
        wolve.sendCommand("boardsize 7")
        winner = run_game(wolve, Qhex, 7, move, args.verbose)
        if(winner == gamestate.PLAYERS["white"]):
            white_wins_Q += 1
        winner = run_game(Qhex, wolve, 7, move, args.verbose)
        if(winner == gamestate.PLAYERS["black"]):
            black_wins_Q += 1
    white_win_rates_Q.append(white_wins_Q/(7*7))
    black_win_rates_Q.append(black_wins_Q/(7*7))
    print("agent Q, opening "+move+", black win rate: "+str(black_wins_Q/(7*7))+", white win rate: "+str(white_wins_Q/(7*7)))

    #Save data for full run
    datafile = 'wolve_learning.save'
    data = {'black_win_rates_Q':black_win_rates_Q, 'white_win_rates_Q':white_win_rates_Q, 'black_win_rates_count':black_win_rates_count, 'white_win_rates_count':white_win_rates_count, 'episodes':episodes}
    with open(datafile, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)



