import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('data_file', type=str, help='File to plot data from')
args = parser.parse_args()

with open(args.data_file, 'rb') as f:
    data = pickle.load(f)
    black_win_rates_solver = np.asarray(data['black_win_rates_solver'])
    white_win_rates_solver = np.asarray(data['white_win_rates_solver'])
    win_rates_solver = (black_win_rates_solver + white_win_rates_solver)/2
    black_win_rates_Q = np.asarray(data['black_win_rates_Q'])
    white_win_rates_Q = np.asarray(data['white_win_rates_Q'])
    win_rates_Q = (black_win_rates_Q + white_win_rates_Q)/2
    black_win_rates_count = np.asarray(data['black_win_rates_count'])
    white_win_rates_count = np.asarray(data['white_win_rates_count'])
    win_rates_count = (black_win_rates_count + white_win_rates_count)/2
    episodes = data['episodes']
    plt.plot(episodes, win_rates_solver, 'r')
    plt.plot(episodes, win_rates_count, 'b')
    plt.plot(episodes, win_rates_Q, 'g')
    plt.ylabel('winrate')
    plt.xlabel('episode')
    plt.show()