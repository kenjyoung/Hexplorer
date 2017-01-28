import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

parser = argparse.ArgumentParser()
parser.add_argument('data_file', type=str, help='File to plot data from')
parser.add_argument('--aggregation', '-a', type=int, default=200, help='Number of episodes to average over.')
args = parser.parse_args()

with open(args.data_file, 'rb') as f:
    data = pickle.load(f)
    Pw_costs = data['Pw_costs']
    Count_costs = data['Count_costs']
    Counts = data['Counts']
    Pw_vars = data['Pw_vars']
    plt.figure(0)
    plt.plot(running_mean(Pw_costs,args.aggregation))
    plt.ylabel('Pw Cost')
    plt.xlabel('Episode')
    plt.draw()
    plt.figure(1)
    plt.plot(running_mean(Count_costs,args.aggregation))
    plt.ylabel('Count Cost')
    plt.xlabel('Episode')
    plt.draw()
    plt.figure(2)
    plt.plot(running_mean(Counts,args.aggregation))
    plt.ylabel('Count')
    plt.xlabel('Episode')
    plt.draw()
    plt.figure(3)
    plt.plot(running_mean(Pw_vars,args.aggregation))
    plt.ylabel('Pw Variance')
    plt.xlabel('Episode')
    plt.show()