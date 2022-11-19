import numpy as np

def add_unknowns_to_one(in_map: np.array, num_out_maps: int):
    """
    in_map should be of shape (column, row, tile)
    """
    columns = in_map.shape[0]
    rows = in_map.shape[1]
    tiles = in_map.shape[2]
    out_whole = np.zeros((columns,rows,tiles+1))
    out_whole[:,:,:-1] = in_map
    
    out_maps = [out_whole]
    for a in range(num_out_maps):
        mask = np.random.uniform(0,1,(columns,rows)) > a/num_out_maps
        unknowns = np.zeros((columns, rows, 1))
        unknowns[mask,:] += 1 
        out = np.append(in_map, unknowns, 2)
        out[mask,:-1] = np.zeros(tiles)
        out_maps.append(out)

    out_maps = np.array(out_maps)
    return out_maps

def add_unknowns(in_maps: np.array, num_out_maps: int):
    out_maps = []
    for map in in_maps:
        out_maps.append(add_unknowns_to_one(map, num_out_maps))

    out_maps = np.concatenate(out_maps)
    return out_maps


if __name__ == "__main__":
    test = np.zeros((10,5,5,3))
    test[:,:,1] += 1

    outs = add_unknowns(test, 2)
    for a, o in enumerate(outs):
        print(a, o)
