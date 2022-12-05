from torch.utils.data import Dataset
import h5py
import numpy as np
from torchvision.transforms import ToTensor

class DomainDataset(Dataset):
    
    def __init__(self, path):
        
        with h5py.File(path, 'r') as f:
            
            self.X = np.array(f["X"])
            self.y = np.array(f["y"])
            self.i = np.array(f["i"])
            
        self.t = ToTensor()
    
    def __getitem__(self, idx):
        
        return self.t(self.X[idx]), self.y[idx], self.i[idx]
    
    def __len__(self):
        
        return self.X.shape[0]