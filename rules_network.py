import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from torch.nn import functional as func
from add_unknowns import add_unknowns

class one_tile_fc(nn.Module):
    def __init__(self, columns, rows, tiles) -> None:
        super().__init__()
        
        in_size = ((columns*rows)-1)*(tiles+1)
        self.top = nn.Sequential(
            nn.Linear(in_size, int(in_size*.75)),
            nn.ReLU(),
            nn.Linear(int(in_size*.75), tiles)
        )

    def forward(self, x):
        outs = self.top(outs)
        return outs

class whole_map_fc(nn.Module):
    def __init__(self, columns, rows, tiles) -> None:
        super().__init__()

        self.columns = columns
        self.rows = rows
        self.tiles = tiles
        self.flatten = nn.Flatten(1,-2) # flatten into array of tile vectors, preserve batch index
        nets = []
        for a in range(columns):
            nets.append([])
            for b in range(rows):
                nets[a].append(one_tile_fc(columns, rows, tiles))

        
    def forward(self, maps):
        """
        maps should come in as nested lists, np arrays or torch tensors with shape (batch, columns, rows, tiles+1)
        """
        if not torch.is_tensor(maps):
            maps  = torch.tensor(maps)

        maps = self.flatten(maps)

        out_probs = torch.empty((self.columns, self.rows, self.tiles))

        
        for a in self.columns:
            offset = a*self.columns
            for b in self.rows:
                if self.training: # if in training, evaluate every tile
                    # remove tile info for the tile the network is inferencing for
                    in_maps = torch.cat((maps[:,:offset+b], maps[:,offset+b+1:]),1) 
                    out_probs[:,a,b] = self.nets[a][b](in_maps)

                else: # if in testing, don't evaluate already filled in tiles
                    map_ids = []
                    in_maps = False
                    for c, m in enumerate(maps):
                        if m[offset+b,-1]: # if this tile is unknown
                            map_ids.append(c) # save the index
                            in_map = torch.cat((m[:offset+b], m[offset+b+1:])) # make the map vector
                            # add it to the array of vectors to infer on
                            if in_maps:
                                in_maps = torch.cat((in_maps,torch.reshape(in_map, [1,*list(in_map.shape)])))
                            else:
                                in_maps = torch.reshape(in_map, [1,*list(in_map.shape)])
                        
                        else: # if the tile is unknown, just return it as is
                            out_probs[c,a,b] = m[offset+b]
                            
                    out_calc = self.nets[a][b](in_maps) # infer on all of the ones we need to

                    for c, m in enumerate(out_calc): # add the inferences back into the fold
                        out_probs[map_ids[c],a,b] = out_calc[c]

        return out_probs



def get_data(path: str):
    pass

def make_data_and_labels(windows):
    pass




if __name__ == "__main__":
    full_windows = get_data("")
    windows = add_unknowns(full_windows)

    model = whole_map_fc()
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(lr=0.001, weight_decay=1e-5)
