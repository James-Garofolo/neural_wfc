import numpy as np

def add_unknowns_to_one(in_map: np.array, num_out_maps: int, max_id: int):
    """
    in_map should be of shape (column, row) containing tile id's
    """
    
    columns = in_map.shape[0]
    rows = in_map.shape[1]
    
    out_maps = [np.copy(in_map)]
    for a in range(num_out_maps):
        mask = np.random.uniform(0,1,(columns,rows)) > a/num_out_maps
        out_map = np.copy(in_map)
        out_map[mask] = max_id + 1
        out_maps.append(out_map)

    out_maps = np.array(out_maps)
    return out_maps

def add_unknowns(in_maps: np.array, num_out_maps: int, max_id: int):
    out_maps = []
    label_maps = []
    for map in in_maps:
        out_maps.append(add_unknowns_to_one(map, num_out_maps, max_id))
        label_maps.append(np.tile(map,[num_out_maps+1,*[1]*len(map.shape)]))

    out_maps = np.concatenate(out_maps)
    label_maps = np.concatenate(label_maps)
    return out_maps, label_maps


if __name__ == "__main__":
    test = np.random.randint(0,10,(10,5,5))

    outs, labels = add_unknowns(test, 3, 9)
    print(outs.shape, labels.shape)
    print('outs: \n', outs[2]) 
    print('labels: \n', labels[2])
