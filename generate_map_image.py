import os, sys, pathlib, math
from types import FunctionType
from PIL import Image, ImageDraw
import numpy as np
from hex_onehot import ONEHOT_LENGTH, onehot_to_hex
from split_map import DIR_TILES_OUTPUT, DIR_DATA, DIR_MAPVECTORS_NP_OUTPUT, TILE_SIZE

tile_sprite_dict = {}

def generate_image(map: np.ndarray, path: str, is_onehot):
	img = Image.new(mode = 'RGB',
		size = (map.shape[0] * TILE_SIZE, map.shape[1] * TILE_SIZE),
		color = (255, 0, 255) # Default to an ugly magenta so we can see if there are any tiles that failed to paste
	)
	for x in range(len(map)):
		column = map[x]
		for y in range(len(column)):
			
			is_unknown_tile = False
			
			if is_onehot:
				# Convert onehot to hexadecimal
				onehot = column[y]
				tile_name = onehot_to_hex(onehot)
			else:
				# Convert index (as int) to hexadecimal string
				tile_name = f'{column[y]:02x}'
				# If the index of the tile is outside the maximum index, that means it's an unknown tile
				is_unknown_tile = (column[y] >= ONEHOT_LENGTH) 
			
			# Skip unknown tiles
			if not is_unknown_tile:
				# If we don't have the sprite in our tile_sprite_dict, load it from the image file
				if (not tile_name in tile_sprite_dict):
					with Image.open(os.path.join(DIR_TILES_OUTPUT, f'{tile_name}.png')) as tile_sprite:
						tile_sprite_dict[tile_name] = tile_sprite.copy()
				coord_x = x * TILE_SIZE
				coord_y = y * TILE_SIZE
				img.paste(tile_sprite_dict[tile_name], (coord_x, coord_y))
	img.save(path, 'PNG')
	
# Generate map image from onehot ndarray
def from_onehot(map_onehot: np.ndarray, path: str):
	return generate_image(map_onehot, path, True)

# Generate map image from ndarray of int indexes
def from_indexes(map_indexes: np.ndarray, path: str):
	return generate_image(map_indexes, path, False)

if __name__ == "__main__":
	
	map_zero = np.load(os.path.join(DIR_MAPVECTORS_NP_OUTPUT, '34.npy'), allow_pickle=True)
	
	print(map_zero.shape)
	cols = []
	for col in map_zero:
		newCol = []
		for row in col:
			thisRow = int(onehot_to_hex(row), 16)
			newCol.append(thisRow)
		cols.append(newCol)
	map_zero_indexes = np.array(cols)
	print(map_zero_indexes.shape)
	
	from_onehot(map_zero, os.path.join(DIR_DATA, 'map0.png'))
	from_indexes(map_zero_indexes, os.path.join(DIR_DATA, 'map0-2.png'))