import sys
sys.path.append("..")
from inputFormat import *

def stateToInput(state):
	board = state.board
	ret = new_game(len(board))
	padding = (input_size - len(board)+1)/2
	for i in range(len(board)):
		for j in range(len(board)):
			if board[i,j] == state.PLAYERS["white"]:
				play_cell(ret, (i+padding,j+padding), white)
			elif board[i,j] == state.PLAYERS["black"]:
				play_cell(ret, (i+padding,j+padding), black)
	return ret