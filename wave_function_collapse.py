import os, sys, pathlib, math, shutil
from types import FunctionType
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import torch
from hex_onehot import ONEHOT_LENGTH, hex_to_onehot
from rules_network import whole_map_fc, conv_window_maker
from split_map import DIR_DATA, DIRNAME, DIR_MAPVECTORS_NP_OUTPUT
import generate_map_image
import random
from sklearn.model_selection import train_test_split # tts for collapse limit
from math import floor, ceil
from consts import MAP_SIZE, IS_MULTIHOT, output_map_size, initial_collapse_limit, collapse_limit_decay, image_output_frequency

class wave_function_collapse:
	
	"""
	
	inputs:
		map_shape: sdfdsf
		num_possible_tiles: sdfsdf
	"""
	def __init__(self, map_shape, num_possible_tiles, collapse_limit = None, guess_multiple = False, limit_decay_rate = 1.0,
				 starting_tiles = None) -> None:
		self.rules = []
		self.undefined_tile = num_possible_tiles # update later
		self.possibilities = np.ones((map_shape[0], map_shape[1], num_possible_tiles))
		self.guess_multiple = guess_multiple
		self.limit_decay = limit_decay_rate
		
		if collapse_limit == None:
			self.collapse_limit = map_shape[0]*map_shape[1]
		else:
			self.collapse_limit = collapse_limit

		if starting_tiles is None:
			self.collapsed_tiles = np.ones(map_shape, dtype=int) * (self.undefined_tile)
		else:
			self.collapsed_tiles = starting_tiles
	
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

		"""
		what up it's jim, i figured out this can be done like this

		entropies = np.sum(self.possibilities, -1) # sum across tile vectors to get entropy calcs
		entropies[self.collapsed_tiles != self.undefined_tile] = self.undefined_tile # make sure filled tiles don't get counted
		lowest_entropy = np.min(entropies) # find lowest entropy
		lowest_coords = np.array(np.where(entropies == lowest_entropy)).T # get entropy coords as pair of column vectors

		and it's faster cause numpy is done in c. left it alone though cause i'm not a fan of jordan code erasure
		"""

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
		for x, y in coords_to_collapse:
			multihot_to_collapse = self.possibilities[x, y]
			where = np.where(multihot_to_collapse == 1)[0]
			# Collapse to a random index within the list of indices that == 1
			# print(self.possibilities[coords_to_collapse])
			
			collapsed_index = where[np.random.randint(0, len(where), )]
			# Set collapsed_tiles to the new tile index at these coords
			self.collapsed_tiles[x, y] = collapsed_index
			# print(f'Collapsed tile {coords} into {collapsed_index:02x}')
		# if len(coords) == 1:
		# 	coords_to_collapse = coords[0]
		# else:
		# 	# Randomly choose a tile to collapse
		# 	index = random.randint(0, len(coords) - 1)
		# 	coords_to_collapse = coords[index]
	
	def add_rule(self, func: FunctionType, idx: int = None):
		"""
		insert a rule into the rule list. use idx to specify a priority index. assumes lowest priority by default
		"""
		if type(func) == FunctionType: # if handler is a function
			if type(idx) == int:
				self.rules.insert(idx, func)
			else:
				self.rules.append(func) # add it to the handler functions
		else: # if not, raise an error before it causes a problem in runtime
			raise TypeError("input to add_press_handler must be of type \"function\"")

	def run_rules(self):
		self.possibilities = np.ones_like(self.possibilities, dtype=np.intc) # start with every tile being possible
		for rule in self.rules: # evaluate each rule
			new_possibilities = np.copy(self.possibilities)
			tile_onehot = rule(self.collapsed_tiles) # eliminate the tiles it says to, leave previously eliminated tiles alone
			new_possibilities &= tile_onehot
			entropies = np.sum(new_possibilities, -1) # sum across tile vectors to get entropy calcs
			lowest_entropy = np.min(entropies) # find lowest entropy

			if lowest_entropy == 0: # if we ran out of options somewhere
				break # don't consider this rule and stop

			elif lowest_entropy == 1: # if we've only got one option somewhere
				self.possibilities = new_possibilities # save that
				break # but don't do more

			else: # if we still have multiple options everywhere
				self.possibilities = new_possibilities # save and keep going

		entropies = np.sum(self.possibilities, -1) # sum across tile vectors to get entropy calcs
		entropies[self.collapsed_tiles != self.undefined_tile] = self.undefined_tile + 1 # make sure filled tiles don't get counted
		lowest_entropy = np.min(entropies) # find lowest entropy
		lowest_coords = np.array(np.where(entropies == lowest_entropy)).T # get entropy coords as pair of column vectors
		# If there are no tiles with entropy of 1, then pick just one randomly
		print(f'Lowest entropy is {lowest_entropy}')
		if (lowest_entropy > 1) and (not self.guess_multiple):
			return [lowest_coords[random.randint(0, len(lowest_coords)-1)]]
		else:
			if len(lowest_coords) > self.collapse_limit:

				self.collapse_limit = max(self.collapse_limit*self.limit_decay,1)
				print("collapse limit:", self.collapse_limit)
				ids = np.arange(len(lowest_coords))
				# fun little trick to shuffle and split the indexes using tts
				ids, _ = train_test_split(ids, train_size=int(self.collapse_limit))
				return lowest_coords[ids]
			else:
				return lowest_coords
		#return [lowest_coords[random.randint(0, len(lowest_coords)-1)]]

	def step_with_rules(self):
		lowest_entropy_coords = self.run_rules()
		if len(lowest_entropy_coords) == 0:
			print('Done')
			return self.collapsed_tiles
		
		self.collapse_state(lowest_entropy_coords)
		
		return self.collapsed_tiles

	def fill(self):
		assert len(self.rules) > 0, "cannot generate map, no rules specified"

		while self.undefined_tile in self.collapsed_tiles:
			self.step_with_rules()

		return self.collapsed_tiles


def make_small_nn_rules(model, model_size, num_tiles, ideal_stride):
	"""
	ideal stride is how many steps you'd like it to take each go if the shape
	worked, the actual stride is going to be rounding-modulated to be close to that
	while fitting the shape of tile_ids
	"""
	def small_nn_rules(tile_ids):
		out_probs = np.zeros([*tile_ids.shape, num_tiles], dtype=np.intc)
		num_steps = [0,0]
		num_steps[0] = int(np.ceil((tile_ids.shape[0]-model_size)/ideal_stride))
		num_steps[1] = int(np.ceil((tile_ids.shape[1]-model_size)/ideal_stride))
		for x in np.round(np.linspace(0, (tile_ids.shape[0]-model_size), num_steps[0])).astype(np.intc):
			for y in np.round(np.linspace(0, (tile_ids.shape[1]-model_size), num_steps[1])).astype(np.intc):
				# take a square from the map and infer on it
				batch = torch.from_numpy(np.array([tile_ids[x:x+model_size, y:y+model_size]], dtype=int))
				nn_prediction = model(batch)[0].detach().numpy()
				threshold = np.mean(nn_prediction, -1)
				threshold = np.moveaxis(np.tile(threshold, (nn_prediction.shape[2], 1, 1)), 0, -1)
				tile_multihot = np.zeros_like(nn_prediction, dtype=np.intc)
				np.greater(nn_prediction,threshold,tile_multihot)
				out_probs[x:x+model_size, y:y+model_size] = tile_multihot
				
		print(out_probs.shape[-1])
		return out_probs

	return small_nn_rules		


def make_single_out_nn_rules(model, model_size, num_tiles, device):
	def single_out_nn_rules(tile_ids):
		out_probs = np.ones([*tile_ids.shape, num_tiles], dtype=np.intc)
		buffer_size = floor(model_size/2)
		
		# batch for pytorch to run all of the inferences at once
		batch = []
		# to keep track of the x, y coordinates for each item in the batch, for looping later
		coordinates = []

		for x in range(tile_ids.shape[0]):
			for y in range(tile_ids.shape[1]):
				if tile_ids[x,y] == num_tiles: # only infer on unknown tiles for speeeed
					# take a square from the map and infer on it
					this_batch_item = np.ones((model_size,model_size), dtype=np.intc) * num_tiles # make window of unknowns
					# this stamps the map onto the window such that a buffer is leftover when necessary
					this_batch_item[max(0, buffer_size-x):min(model_size, buffer_size+tile_ids.shape[0]-x),\
						max(0, buffer_size-y):min(model_size, buffer_size+tile_ids.shape[1]-y)] = \
							tile_ids[max(0,x-buffer_size):min(x+buffer_size+1,tile_ids.shape[0]),\
							max(0,y-buffer_size):min(y+buffer_size+1,tile_ids.shape[1])].astype(np.intc)
					
					batch.append(this_batch_item)
					coordinates.append((x, y))
		
		np_batch = np.array(batch)
		batch = torch.from_numpy(np_batch).to(device)
		nn_prediction = model(batch).cpu().detach().numpy()
		i = 0 # for the crashdump
		# Process the predictions
		for prediction, coords in zip(nn_prediction, coordinates):
			x, y = coords
			
			threshold = np.mean(prediction, -1)
			tile_multihot = np.zeros_like(prediction, dtype=np.intc)
			tile_multihot[prediction>threshold] = 1
			if np.sum(tile_multihot) == 0:
				print('UH OH PANIC MODE')
				print(f'x, y = {coords}, i={i}')
				np.save('data/batch.npy', np_batch, allow_pickle=True)
				np.save('data/probably-the-right-batch-index.npy', np_batch[i], allow_pickle=True)
				np.save('data/prediction.npy', prediction, allow_pickle=True)
				exit()

			out_probs[x,y] = tile_multihot
			i += 1
	
		#print(np.sum(out_probs, -1))
		return out_probs

	return single_out_nn_rules	

if __name__ == '__main__':
	# remove and recreate output folder
	path = os.path.join(DIR_DATA, 'steps')
	shutil.rmtree(path)
	pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
	
	print(f'Using model trained on {MAP_SIZE}x{MAP_SIZE}')
	
	# Get the shape of the maps
	#map_zero = np.load(os.path.join(DIR_MAPVECTORS_NP_OUTPUT, '0.npy'), allow_pickle=True)
	
	tile_vector_length = 90
	#start_tiles = np.ones((256,88), dtype=int) * tile_vector_length
	#start_tiles[0,:] = 45
	#start_tiles[-1,:] = 45
	#start_tiles[:,0] = 45
	#start_tiles[:,-1] = 45

	wfc = wave_function_collapse(output_map_size, tile_vector_length, collapse_limit=initial_collapse_limit, guess_multiple=True, limit_decay_rate=collapse_limit_decay)
	print(tile_vector_length)
	# Load the PyTorch model
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")

	if IS_MULTIHOT:
		model_file = os.path.join(DIRNAME, f'rules_gen_{MAP_SIZE}_1_out_multihot.pt')
	else:
		model_file = os.path.join(DIRNAME, f'rules_gen_{MAP_SIZE}_1_out.pt')
	with open(model_file, 'rb') as f:
		model: conv_window_maker = torch.load(f, map_location=torch.device(device))
	
	model.to(device) # just run on cpu to keep things simpler

	"""def nn_rules(tile_ids):
		batch = torch.from_numpy(np.array([tile_ids], dtype=int))
		nn_prediction = model(batch)[0].detach().numpy()
		threshold = np.mean(nn_prediction, -1)
		threshold = np.moveaxis(np.tile(threshold, (nn_prediction.shape[2], 1, 1)), 0, -1)
		tile_multihot = np.zeros_like(nn_prediction, dtype=np.intc)
		np.greater(nn_prediction,threshold,tile_multihot)
		return tile_multihot"""
			
	#wfc.add_rule(make_small_nn_rules(model, 7, tile_vector_length, ideal_stride=3))
	wfc.add_rule(make_single_out_nn_rules(model, MAP_SIZE, tile_vector_length, device))

	PATH_TILE = 1 # index of the walkable path tile
	
	# Get the initial board state
	collapsed_tiles = wfc.first_step()
	
	#collapsed_tiles[0, 0] = 90 # manually insert a path tile in the left edge
	step = 0
	while ONEHOT_LENGTH in collapsed_tiles:
		print(f'\nSTEP {step}')

		"""batch = torch.from_numpy(np.array([collapsed_tiles], dtype=int))
		
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
		collapsed_tiles = wfc.step(tile_multihot)"""
		
		collapsed_tiles = wfc.step_with_rules()
		
		# Generate an image so we can visualize what the current map looks like
		if step % image_output_frequency == 0:
			generate_map_image.from_indexes(collapsed_tiles, os.path.join(DIR_DATA, f'steps/step_{step}.png'))
		step += 1
	
	generate_map_image.from_indexes(collapsed_tiles, os.path.join(DIR_DATA, f'steps/complete.png'))
	
	while True:
		try:
			os.remove(os.path.join(DIR_DATA, f'steps/step_{step}.png'))
			step += 1
		except FileNotFoundError:
			break
