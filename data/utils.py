import numpy as np 
import torch.utils.data as d

class Dataset(d.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X, self.y = X, y 
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return np.array(self.X[i]), self.y[i]


def load_data(X, y, batch_size = 1):
    '''
    Returns the data loader for provided feature matrix X and targets y
    '''
    data = Dataset(X, y)
    loader = d.DataLoader(data, batch_size = batch_size)
    return loader

