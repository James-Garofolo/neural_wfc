import os, sys, pathlib, math
from types import FunctionType
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import torch
from hex_onehot import ONEHOT_LENGTH, hex_to_onehot
from rules_network import whole_map_fc, get_data
from split_map import DIR_DATA, DIRNAME, DIR_MAPVECTORS_NP_OUTPUT
import generate_map_image
import random

class wave_function_collapse:
	
	"""
	
	inputs:
		map_shape: sdfdsf
		num_possible_tiles: sdfsdf
	"""
	def __init__(self, map_shape, num_possible_tiles) -> None:
		self.manual_rules = []
		self.undefined_tile = num_possible_tiles # update later
		self.possibilities = np.zeros((map_shape[0], map_shape[1], num_possible_tiles))
		self.collapsed_tiles = np.ones(map_shape, dtype=int) * (self.undefined_tile)
	
	"""
	Returns the first state of the board
	"""
	def first_step(self) -> np.ndarray:
		return self.collapsed_tiles
	
	""" 
	Calculates and returns the next state of the board
	
	"""
	def step(self, new_possibilities: np.ndarray) -> np.ndarray:
		self.possibilities = new_possibilities
		
		lowest_entropy_coords = self.find_lowest_entropy()
		if len(lowest_entropy_coords) == 0:
			print('Done')
			return self.collapsed_tiles
		
		self.collapse_state(lowest_entropy_coords)
		
		return self.collapsed_tiles
	
	"""
	Finds the tile(s) with the lowest entropy
	Returns the x and y indices of those tiles
	"""
	def find_lowest_entropy(self):
		lowest_entropy = self.undefined_tile + 1 # bigger than the highest possible entropy
		lowest_coords = []
		for y in range(len(self.possibilities)):
			column: np.ndarray = self.possibilities[y]
			for x in range(len(column)):
				if self.collapsed_tiles[y,x] == self.undefined_tile:
					this_multihot: np.ndarray = column[x]
					# Find the indices where this multihot is 1 / true
					where = np.where(this_multihot == 1)[0]
					this_entropy = len(where)
					#print(this_entropy)
					# if this is the new lowest, reset lowest_coords
					if (this_entropy > 0) and (this_entropy < lowest_entropy):
						lowest_coords = [(y, x)]
						lowest_entropy = this_entropy
					# if this has the same entropy as lowest, simply append
					elif this_entropy == lowest_entropy:
						lowest_coords.append((y, x))
		# If there are no tiles with entropy of 1, then pick just one randomly
		print(f'Lowest entropy is {lowest_entropy}')
		if lowest_entropy > 1:
			return [lowest_coords[random.randint(0, len(lowest_coords)-1)]]
		else:
			return lowest_coords
	
	"""
	Collapse the state of a given tile, or randomly choose between 
	multiple tiles and collapse the state of that tile
	"""
	def collapse_state(self, coords_to_collapse):
		print(f'Collapsing {len(coords_to_collapse)} tiles')
		for coords in coords_to_collapse:
			multihot_to_collapse = self.possibilities[coords]
			where = np.where(multihot_to_collapse == 1)[0]
			# Collapse to a random index within the list of indices that == 1
			# print(self.possibilities[coords_to_collapse])
			
			collapsed_index = where[random.randint(0, len(where) - 1)]
			# Set collapsed_tiles to the new tile index at these coords
			self.collapsed_tiles[coords] = collapsed_index
			# print(f'Collapsed tile {coords} into {collapsed_index:02x}')
		# if len(coords) == 1:
		# 	coords_to_collapse = coords[0]
		# else:
		# 	# Randomly choose a tile to collapse
		# 	index = random.randint(0, len(coords) - 1)
		# 	coords_to_collapse = coords[index]
	
	def add_manual_rule(self, name: str, func: FunctionType):
		pass


if __name__ == '__main__':	
	# Get the shape of the maps
	map_zero = np.load(os.path.join(DIR_MAPVECTORS_NP_OUTPUT, '0.npy'), allow_pickle=True)
	
	wfc = wave_function_collapse(map_zero.shape, ONEHOT_LENGTH)
	
	# Load the PyTorch model
	model_file = os.path.join(DIRNAME, 'rules_gen_fc_no_dupes.pt')
	with open(model_file, 'rb') as f:
		model: whole_map_fc = torch.load(f, map_location=torch.device('cpu'))
	
	model.to('cpu') # just run on cpu to keep things simpler
	
	PATH_TILE = 1 # index of the walkable path tile
	
	# Get the initial board state
	collapsed_tiles = wfc.first_step()
	
	#collapsed_tiles[0, 0] = 90 # manually insert a path tile in the left edge
	step = 0
	while 89 in collapsed_tiles:
		print(f'\nSTEP {step}')
		batch = torch.from_numpy(np.array([collapsed_tiles], dtype=int))
		
		#print(batch)
		#print(batch.shape)
		# Make prediction
		nn_prediction = model(batch)[0].detach().numpy()
		# print('Prediction:', nn_prediction[3, 0]) 
		
		
		threshold = np.mean(nn_prediction, -1)
		threshold = np.moveaxis(np.tile(threshold, (nn_prediction.shape[2], 1, 1)), 0, -1)
		tile_multihot = np.zeros_like(nn_prediction)
		np.greater(nn_prediction,threshold,tile_multihot)

		
		
		# Perform WFC computations
		collapsed_tiles = wfc.step(tile_multihot)
		"""for a, column in enumerate(collapsed_tiles):
			for b, val in enumerate(column):
				if val != 89:
					print(nn_prediction[a,b])"""
		
		#1/0
		# print('Collapsed tiles:', collapsed_tiles[3, 0])
		
		# Generate an image so we can visualize what the current map looks like
		generate_map_image.from_indexes(collapsed_tiles, os.path.join(DIR_DATA, f'step_{step}.png'))
		step += 1