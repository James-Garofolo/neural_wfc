import numpy as np

# def zelda_hex_to_our_hex(hex: str):
# 	thisInt = int(hex, 16)
# 	ret = thisInt
# 	for i in [0x11, 0x25, 0x39, 0x4d, 0x61, 0x75, 0x89, 0x9d]:
# 		if thisInt > i:
# 			ret = ret - 2
# 	return f'{ret:02x}'

# length of the onehot vector
ONEHOT_LENGTH = int('8f', 16)

# Convert a zelda tile hex string (2 characters) to its corresponding NumPy onehot
def hex_to_onehot(hex: str):
	onehot = np.zeros(ONEHOT_LENGTH)
	thisInt = int(hex, 16) - 1 # 0-based
	onehot[thisInt] = 1
	return onehot

# Conver a NumPy onehot to its corresponding zelda tile hex string
def onehot_to_hex(onehot: np.ndarray):
	where = np.where(onehot == 1)[0]
	# make sure there is exactly one instance where onehot == 1
	if not len(where) == 1:
		raise RuntimeError(f'Onehot is not onehot! np.where(onehot == 1)[0] = {where}')
	thisInt = where[0] + 1
	return f'{thisInt:02x}'

def multihot_to_hex(multihot: np.ndarray):
	where = np.where(multihot == 1)[0]
	if len(where) == 0:
		raise RuntimeError(f'Multihot is all zeros! np.where(multhot == 1)[0] = {where}')
	ret = []
	for thisInt in where[0]:
		ret.append(f'{thisInt:02x}')
	return np.ndarray(x, dtype=str)

def hex_to_multihot(hexes: np.ndarray):
	pass

# tests
if __name__ == "__main__":
	print(onehot_to_hex(hex_to_onehot('01')))
	print(onehot_to_hex(hex_to_onehot('05')))
	print(onehot_to_hex(hex_to_onehot('20')))
	print(onehot_to_hex(hex_to_onehot('7a')))
	print(onehot_to_hex(hex_to_onehot('9d')))
	print(zelda_hex_to_our_hex('14'))
	print(zelda_hex_to_our_hex('28'))