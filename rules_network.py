from torch.nn.modules.activation import LogSoftmax
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from torch.nn import functional as func
import os
import math
from add_unknowns import add_unknowns

"""
change fc network guy to use embeddings with index inputs rather than one-hot vector inputs
"""


class whole_map_fc(nn.Module):
    def __init__(self, columns, rows, tiles, hidden_ratio=1.25) -> None:
        super().__init__()

        self.columns = columns
        self.rows = rows
        self.tiles = tiles
        #print('saved dims')
        embedding_dim = int(self.tiles/10)
        print("embedding dims:", embedding_dim)
        self.embed = nn.Embedding(self.tiles+1, embedding_dim)
        self.flatten = nn.Flatten() # flatten into array of tile vectors, preserve batch index
        #print('made special layers')
        
        self.top = nn.Sequential(
            nn.Linear(columns*rows*embedding_dim, int(columns*rows*embedding_dim*hidden_ratio)),
            nn.ReLU(),
            nn.Linear(int(columns*rows*embedding_dim*hidden_ratio), columns*rows*(tiles-1)),
            nn.ReLU()
        )

        self.softmax = nn.Softmax(3)
        self.logify = nn.Sigmoid()

        
    def forward(self, maps):
        """
        maps should come in as nested lists, np arrays or torch tensors with shape (batch, columns, rows, tiles+1)
        """
        if not torch.is_tensor(maps):
            maps  = torch.tensor(maps)

        #print("in shape:", maps.shape, torch.max(maps))
        maps = self.embed(maps)
        #print("embedded:", maps.shape)
        maps = self.flatten(maps)
        #print("flattened:", maps.shape)
        out_probs = self.top(maps)
        #print("inferred", out_probs.shape)
        #outs = self.softmax(out_probs.view(-1,self.columns,self.rows,self.tiles-1))
        outs = self.logify(out_probs.view(-1,self.columns,self.rows,self.tiles-1))
        #print("softmaxed", outs.shape, torch.max(outs))


        return outs



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


def idx_to_one_hot(ids: np.array, max_id):
    outs = np.zeros((ids.shape[0], ids.shape[1], ids.shape[2], max_id), dtype=np.single)
    for a, map in enumerate(ids):
        for b, row in enumerate(map):
            for c, id in enumerate(row):
                outs[a, b, c, id] = 1.00

    return outs



def train(data, labels, model, device, loss_fn, optimizer, verbose=True, batch_size=64):
    model.train()

    avg_loss = 0
    counter = 0
    correct = 0
    classified = 0

    size = data.shape[0]
    batches = int(np.ceil(size/batch_size))

    if not torch.is_tensor(data):
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)

    for batch in range(batches):
        try:
            X, y = data[batch*batch_size:(batch+1)*batch_size], labels[batch*batch_size:(batch+1)*batch_size]

        except IndexError:
            X, y = data[batch*batch_size:-1], labels[batch*batch_size:-1]

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        
        loss = loss_fn(pred, y)
        tiles = torch.argmax(pred, dim=3)
        label_tiles = torch.argmax(y, dim=3)
        
        correct += (tiles==label_tiles).type(torch.float).sum().item()
        classified += torch.numel(tiles)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if verbose:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"acc: {correct/classified}")
            avg_loss += loss
            counter += 1

    return correct/classified, avg_loss/counter


def test(data, labels, model, device, loss_fn):
    correct = 0
    classified = 0
    if not torch.is_tensor(data):
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)

    model.eval()
    size = data.shape[0]
    with torch.no_grad():
        X = data
        y = labels
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss = loss_fn(pred, y).item()
        test_loss /= size
        
        tiles = torch.argmax(pred, dim=3)
        label_tiles = torch.argmax(y, dim=3)
        
        correct += (tiles==label_tiles).type(torch.float).sum().item()
        classified += torch.numel(tiles)
    print(f"Avg loss: {test_loss:>8f} \n")
    print(f"Avg acc: {correct/classified}\n")
    return correct/classified, test_loss


if __name__ == "__main__":
    full_windows, max_id = get_data(os.getcwd() + "/data/map_vectors/numpy/")
    print("data in: ", full_windows.shape)
    data_windows, label_windows = add_unknowns(full_windows, 100, max_id)
    print("with unknowns added: ", data_windows.shape, label_windows.shape)
    print(max_id)

    train_data, val_data, train_labels, val_labels = train_test_split(data_windows, label_windows, test_size=0.1)
    
    train_labels = idx_to_one_hot(train_labels, max_id)
    val_labels = idx_to_one_hot(val_labels, max_id)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = whole_map_fc(label_windows.shape[1], label_windows.shape[2], max_id+1, 0.5)
    model.to(device)
    loss = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    print('starting to train')

    best_loss = None
    best_acc = 0
    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_acc, train_loss = train(train_data, train_labels, model, device, loss, optim, False)
        test_acc, test_loss = test(val_data, val_labels, model, device, loss)
        if (best_loss == None) or (best_loss > test_loss) or (test_acc > best_acc):
            best_loss = test_loss
            best_acc = test_acc
            with open('rules_gen_fc_long.pt', 'wb') as f:
                torch.save(model, f)
        
        else:
            optim.defaults['lr'] /= 2
        
    print("Done!")