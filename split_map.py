import os, sys, pathlib, math
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from hex_onehot import ONEHOT_LENGTH, hex_to_onehot

# number of tiles in the overworld
NUM_MAPS_X = 16
NUM_MAPS_Y = 8

# x/y size of each square tile, in px
TILE_SIZE = 16
# the map image thingy cuts off the bottom 8 pixels of each tile
MAP_OVERFLOW_Y = 8
# number of tiles per map
MAP_SIZE_X = 16
MAP_SIZE_Y = 11

DIRNAME = os.path.dirname(__file__)
DIR_DATASRC = os.path.join(DIRNAME, 'data_source')
DIR_DATA = os.path.join(DIRNAME, 'data')
DIR_MAP_OUTPUT = os.path.join(DIR_DATA, 'map')
DIR_TILES_OUTPUT = os.path.join(DIR_MAP_OUTPUT, 'tiles')
DIR_MAPVECTORS_OUTPUT = os.path.join(DIR_DATA, 'map_vectors')
DIR_MAPVECTORS_CSV_OUTPUT = os.path.join(DIR_MAPVECTORS_OUTPUT, 'csv')
DIR_MAPVECTORS_NP_OUTPUT = os.path.join(DIR_MAPVECTORS_OUTPUT, 'numpy')


def main():
	# create output path(s)
	pathlib.Path(DIR_TILES_OUTPUT).mkdir(parents=True, exist_ok=True) 
	pathlib.Path(DIR_MAPVECTORS_CSV_OUTPUT).mkdir(parents=True, exist_ok=True) 
	pathlib.Path(DIR_MAPVECTORS_NP_OUTPUT).mkdir(parents=True, exist_ok=True) 
	
	tiles_csv_path = os.path.join(DIR_DATASRC, 'zelda_overworld_tiles.csv')
	img_path = os.path.join(DIR_DATASRC, 'zelda_overworld_image.png')
	
	csv_df = pd.read_csv(tiles_csv_path, header=None).astype(str).transpose()
	
	all_tile_names = csv_df.stack().unique()
	all_tile_names.sort()
	# print(len(all_tile_names))
	
	tile_sprite_dict = {}
	
	print('Opening source image')
	
	with Image.open(img_path) as img:
		if (csv_df.shape[0] * TILE_SIZE != img.size[0] or (csv_df.shape[1] * TILE_SIZE - MAP_OVERFLOW_Y * NUM_MAPS_Y) != img.size[1]):
			raise RuntimeError('CSV data does not line up with the map image data!')
		
		print('Extracting tile sprites')
		# Loop through the dataframe & cut a tile sprite for each of the tiles
		for y, row in csv_df.items():
			for x in range(len(row)):
				tile_name = row[x]
				if (not tile_name in tile_sprite_dict):
					this_map_row = math.floor(y / MAP_SIZE_Y)
					# Crop the tile sprite from the main image & save it to the dict object
					coord_x = x * TILE_SIZE
					coord_y = y * TILE_SIZE - (this_map_row * MAP_OVERFLOW_Y)
					box = (coord_x, coord_y, coord_x + TILE_SIZE, coord_y + TILE_SIZE)
					tile_sprite_dict[tile_name] = img.crop(box)
		
		# Sanity check to make sure we got a sprite for each tile
		for tile_name in all_tile_names:
			if (not tile_name in tile_sprite_dict):
				raise RuntimeError(f'Did not find a sprite for tile type: {tile_name}')
		
		print('Outputting individual tile sprites as PNG')
		# Save the tiles
		for tile_name in tile_sprite_dict:
			this_tile: Image = tile_sprite_dict[tile_name]
			this_tile.save(os.path.join(DIR_TILES_OUTPUT, f'{tile_name}.png'), 'PNG')
	
	print('Creating complete map image')
	# Recreate a complete map from our new tile sprites
	img = Image.new(mode = 'RGB', 
		size = (csv_df.shape[0] * TILE_SIZE, csv_df.shape[1] * TILE_SIZE),
		color = (255, 0, 255))
	
	for y, row in csv_df.items():
		for x in range(len(row)):
			# Place each sprite in order
			tile_name = row[x]
			coord_x = x * TILE_SIZE
			coord_y = y * TILE_SIZE
			img.paste(tile_sprite_dict[tile_name], (coord_x, coord_y))
			
	img.save(os.path.join(DIR_MAP_OUTPUT, 'complete.png'), 'PNG')
	
	# Create individual map images
	# for map_y in range(NUM_MAPS_Y):
	# 	for map_x in range(NUM_MAPS_X):
	# 		this_idx = map_x + map_y * NUM_MAPS_X
	# 		# Crop this map from the main image we assembled
	# 		coord_x = map_x * NUM_MAPS_X * TILE_SIZE
	# 		coord_y = map_y * NUM_MAPS_Y * TILE_SIZE
	# 		box = (coord_x, coord_y, coord_x + MAP_SIZE_X * TILE_SIZE, coord_y + MAP_SIZE_Y * TILE_SIZE)
	# 		this_map = img.crop(box)
	# 		this_map.save(os.path.join(DIR_MAP_OUTPUT, f'{this_idx}.png'), 'PNG')
	
	
	print('Creating individual map files')
	for map_y in range(NUM_MAPS_Y):
		for map_x in range(NUM_MAPS_X):
			this_idx = map_x + map_y * NUM_MAPS_X
			# split up the dataframe into the chunk associated with this map
			this_rangeIdx = csv_df.columns[map_y * MAP_SIZE_Y : (map_y + 1) * MAP_SIZE_Y]
			this_mapColumn = csv_df[this_rangeIdx]
			this_range = this_mapColumn[map_x * MAP_SIZE_X : (map_x + 1) * MAP_SIZE_X]
			this_range = pd.DataFrame(this_range.to_numpy()) # force reset the axes to 0
			
			# Create an image for this map
			this_map = Image.new(mode = 'RGB', 
				size = (MAP_SIZE_X * TILE_SIZE, MAP_SIZE_Y * TILE_SIZE),
				color = (255, 0, 255))
			
			for y, row in this_range.items():
				for x in range(len(row)):
					tile_name = row[x]
					# Place each sprite in order
					coord_x = x * TILE_SIZE
					coord_y = y * TILE_SIZE
					this_map.paste(tile_sprite_dict[tile_name], (coord_x, coord_y))
			# Save the image
			this_map.save(os.path.join(DIR_MAP_OUTPUT, f'{this_idx}.png'), 'PNG')
			
			# Convert into one-hot vectors
			onehot = this_range.apply(lambda x: x.astype(str).map(hex_to_onehot))
			np.save(os.path.join(DIR_MAPVECTORS_NP_OUTPUT, f'{this_idx}.npy'), onehot.to_numpy(), allow_pickle=True)
			onehot.to_csv(os.path.join(DIR_MAPVECTORS_CSV_OUTPUT, f'{this_idx}.csv'), header=None, index=None)
			print(f'  {this_idx+1} of {NUM_MAPS_X * NUM_MAPS_Y}   ', end='\r')
	
	notes = ''
	for tile_name in all_tile_names:
		onehot = hex_to_onehot(tile_name).astype(int)
		notes = notes + f'{tile_name}: {np.array2string(onehot, max_line_width=1000, precision=None)} \n'
	
	with open(os.path.join(DIR_MAPVECTORS_OUTPUT, 'notes.txt'), 'w') as f:
		f.write(notes)
	
	print('\nDone')
			

if __name__ == "__main__":
	main()