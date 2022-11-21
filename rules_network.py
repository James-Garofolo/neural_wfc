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
import os

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
    """
    getting in one 2d array of 1d one-hot vectors, need to open for each file and turn them into
    """
    maps = []
    for a in range(128):
        map = np.load(f'{path}/{a}.npy', allow_pickle=True)
        columns = []
        length = 0
        for column in map:
            rows = []
            for row in column:
                if length == 0:
                    length = row.size
                rows.append(np.argmax(row))
            
            columns.append(np.array(rows))
        maps.append(np.stack(columns))
    maps = np.stack(maps)
    return maps, length


def train(data, labels, model, device, loss_fn, optimizer, batch_size=64):
    model.train()

    avg_loss = 0
    counter = 0

    size = data.shape[0]
    batches = np.ceil(size/batch_size)

    for batch in range(batches):
        try:
            X, y = data[batch*batch_size:(batch+1)*batch_size], labels[batch*batch_size:(batch+1)*batch_size]

        except IndexError:
            X, y = data[batch*batch_size:-1], labels[batch*batch_size:-1]

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            avg_loss += loss
            counter += 1

    return avg_loss/counter


def test(data, labels, model, device, loss_fn):
    
    model.eval()
    size = data.shape[0]
    with torch.no_grad():
        X = data
        y = labels
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss = loss_fn(pred, y).item()
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct, loss


if __name__ == "__main__":
    full_windows, max_id = get_data(os.getcwd() + "/data/map_vectors/numpy/")
    print("data in: ", full_windows.shape)
    data_windows, label_windows = add_unknowns(full_windows, 20, max_id)
    print("with unknowns added: ", data_windows.shape, label_windows.shape)

    train_data, val_data, train_labels, val_labels = train_test_split(data_windows, label_windows, test_size=0.1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = whole_map_fc(label_windows.shape[1], label_windows.shape[2], label_windows.shape[3])
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(lr=0.001, weight_decay=1e-5)


    best_loss = None
    best_acc = 0
    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_data, train_labels, model, device, loss, optim)
        test_acc, test_loss = test(val_data, val_labels, model, device, loss)
        if (best_loss == None) or (best_loss > test_loss) or (best_acc < test_acc):
            best_acc = test_acc
            best_loss = test_loss
            with open('rules_gen_fc.pt', 'wb') as f:
                torch.save(model, f)
        
        else:
            optim.defaults['lr'] /= 2
        
    print("Done!")