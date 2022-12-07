import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import os

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
            nn.Linear(int(columns*rows*embedding_dim*hidden_ratio), columns*rows*tiles),
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
        outs = self.logify(out_probs.view(-1,self.columns,self.rows,self.tiles))
        #outs = out_probs.view(-1,self.columns,self.rows,self.tiles)
        #print("softmaxed", outs.shape, torch.max(outs))


        return outs

def get_data_onehots(path: str):
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
                    length = row.size-1
                rows.append(np.argmax(row))
            
            columns.append(np.array(rows))
        maps.append(np.stack(columns))
    maps = np.stack(maps)
    return maps, length

def get_data_ids(path: str, num_windows: int = 128):
    """
    getting in one 2d array of 1d one-hot vectors, need to open for each file and turn them into
    """
    maps = []
    for a in range(9,num_windows, 15):
        map = np.load(f'{path}/{a}.npy', allow_pickle=True)
        maps.append(map)
    maps = np.stack(maps)
    length = np.max(maps)
    return maps, length

def idx_to_one_hot(ids: np.array, max_id):
    outs = np.zeros((ids.shape[0], ids.shape[1], ids.shape[2], max_id+1), dtype=np.intc)
    for a, map in enumerate(ids):
        for b, row in enumerate(map):
            for c, id in enumerate(row):
                outs[a, b, c, id] = 1.00

    return outs


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
    overlap_count = 0
    for a, map in enumerate(in_maps):
        if a%100 == 0:
            print(f"   map: {a}   ")
        #out_maps.append(add_unknowns_to_one(map, num_out_maps, max_id))
        out_map = add_unknowns_to_one(map, num_out_maps, max_id)
        label_maps.append(idx_to_one_hot(np.tile(map,[num_out_maps+1,*[1]*len(map.shape)]), max_id))
        for a, sample in enumerate(out_map):
            for b, other_map in enumerate(out_maps):
                if np.array_equal(sample, other_map[a]):
                    label_maps[b][a] |= label_maps[-1][a]
                    label_maps[-1][a] = label_maps[b][a]
                    overlap_count += 1
        out_maps.append(out_map)

    print(f"found {overlap_count} overlaps")
    out_maps = np.concatenate(out_maps)
    label_maps = np.concatenate(label_maps).astype(np.single)
    return out_maps, label_maps


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
        pred = pred.cpu().detach().numpy()
        threshold = np.mean(pred, -1)
        threshold = np.moveaxis(np.tile(threshold, (pred.shape[-1],*[1]*(len(pred.shape)-1))), 0, -1)
        tile_multihot = np.zeros_like(pred, dtype=np.intc)
        np.greater(pred,threshold,tile_multihot)
        label_tiles = y.cpu().detach().numpy().astype(np.intc)

        correct_tiles = ~(tile_multihot ^ label_tiles)&1 # set to 1 if both are same
        
        correct += np.sum(correct_tiles)
        classified += correct_tiles.size

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
        
        pred = pred.cpu().detach().numpy()
        threshold = np.mean(pred, -1)
        threshold = np.moveaxis(np.tile(threshold, (pred.shape[-1],*[1]*(len(pred.shape)-1))), 0, -1)
        tile_multihot = np.zeros_like(pred, dtype=np.intc)
        np.greater(pred,threshold,tile_multihot)
        label_tiles = y.cpu().detach().numpy().astype(np.intc)
        
        correct_tiles = ~(tile_multihot ^ label_tiles)&1 # set to 1 if both are same
        
        correct += np.sum(correct_tiles)
        classified += correct_tiles.size

    print(f"Avg loss: {test_loss:>8f} \n")
    print(f"Avg acc: {correct/classified}\n")
    return correct/classified, test_loss


if __name__ == "__main__":
    full_windows, max_id = get_data_ids(os.getcwd() + "/data/map_vectors/numpy", 20500)
    print("data in: ", full_windows.shape)
    data_windows, label_windows = add_unknowns(full_windows, 100, max_id)
    print("with unknowns added: ", data_windows.shape, label_windows.shape)
    print("max id:", max_id)

    train_data, val_data, train_labels, val_labels = train_test_split(data_windows, label_windows, test_size=0.1)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    with open("rules_gen_7.pt", 'rb') as f:
        model: whole_map_fc = torch.load(f, map_location=torch.device('cpu'))
    #model = whole_map_fc(label_windows.shape[1], label_windows.shape[2], max_id+1, 0.5).to(device)
    loss = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    print('starting to train')

    best_loss = None
    best_acc = 0
    epochs = 2
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_acc, train_loss = train(train_data, train_labels, model, device, loss, optim, False)
        test_acc, test_loss = test(val_data, val_labels, model, device, loss)
        if (best_loss == None) or (best_loss > test_loss) or (test_acc > best_acc):
            best_loss = test_loss
            best_acc = test_acc
            with open('rules_gen_7.pt', 'wb') as f:
                torch.save(model, f)
        
        else:
            optim.defaults['lr'] /= 2
        
    print("Done!")