import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

parser = argparse.ArgumentParser()
parser.add_argument('data_file', type=str, help='File to plot data from')
parser.add_argument('--aggregation', '-a', type=int, default=1, help='Number of episodes to average over.')
args = parser.parse_args()

with open(args.data_file, 'rb') as f:
    data = pickle.load(f)
    black_win_rates_Q = data['black_win_rates_Q']
    white_win_rates_Q = data['white_win_rates_Q']
    black_win_rates_count = data['black_win_rates_count']
    white_win_rates_count = data['white_win_rates_count']
    temps = data['temp']
    plt.figure(0)
    plt.plot(running_mean(Pw_costs,args.aggregation))
    plt.ylabel('temp')
    plt.xlabel('Episode')
    plt.draw()
    plt.figure(1)
    plt.plot(running_mean(Qsigma_costs,args.aggregation))
    plt.ylabel('Qsigma Cost')
    plt.xlabel('Episode')
    plt.draw()
    plt.figure(2)
    plt.plot(running_mean(Qsigmas,args.aggregation))
    plt.ylabel('Qsigma')
    plt.xlabel('Episode')
    plt.draw()
    plt.figure(3)
    plt.plot(running_mean(Pw_vars,args.aggregation))
    plt.ylabel('Pw Variance')
    plt.xlabel('Episode')
    plt.show()