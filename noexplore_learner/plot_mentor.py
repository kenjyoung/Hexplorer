import pickle
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('data_file', type=str, help='file to plot data from')
args = parser.parse_args()

with open(args.data_file, 'rb') as f:
    data = pickle.load(f)
    Pw_costs = data['Pw_costs']
    plt.figure(0)
    plt.plot(Pw_costs)
    plt.ylabel('Pw Cost')
    plt.xlabel('Episode')
    plt.draw()
    plt.show()

