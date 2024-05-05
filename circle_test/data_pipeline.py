import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Convert data to tensors
class Data(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

def prepare_dataset():
    batch_size = 64
    # Create a dataset with 10,000 samples.
    x, y = make_circles(n_samples = 10000,
                        noise= 0.05,
                        random_state=26)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=26)

    # instantiate training and test data
    train_dataset = Data(x_train, y_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle= True)

    test_dataset = Data(x_test, y_test)
    test_dataloader = DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle= True)

    return train_dataloader, test_dataloader




