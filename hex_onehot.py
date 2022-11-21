import numpy as np
# length of the onehot vector
ONEHOT_LENGTH = int('9d', 16)

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

# tests
if __name__ == "__main__":
	print(onehot_to_hex(hex_to_onehot('01')))
	print(onehot_to_hex(hex_to_onehot('05')))
	print(onehot_to_hex(hex_to_onehot('20')))
	print(onehot_to_hex(hex_to_onehot('7a')))
	print(onehot_to_hex(hex_to_onehot('9d')))