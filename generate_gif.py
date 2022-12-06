import os, sys, pathlib, math
from PIL import Image, ImageDraw
import numpy as np
from split_map import DIRNAME, DIR_DATA

# Things that can be changed
filename_base = '' # if the filename is #.png, keep as an empty string; 
start = 0
end = 20
num_zeros = 0 # if greater than zero, then this is the # of zeros to pad the filenames
file_ext = '.png'
output_filename = 'out.gif'

path_base = os.path.join(DIR_DATA, 'map', filename_base)
output_path = os.path.join(DIR_DATA, output_filename)

paths = np.arange(start, end).astype(str)
# pad zeros if desired
if num_zeros > 0:
	paths = np.char.rjust(paths, num_zeros, '0')
paths = np.char.add(path_base, paths)
paths = np.char.add(paths, file_ext)

images = []
for path in paths:
	try:
		with Image.open(path) as src_img:
			img = Image.new('RGB', src_img.size)
			img.paste(src_img)
			images.append(img)
	except OSError as err:
		print(err)
		exit()

images[0].save(output_path, save_all=True, append_images=images[1:], optimize=True, duration=800, loop=0,)