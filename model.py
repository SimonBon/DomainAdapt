from torch import nn
import torch

class DownBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, dtype=float):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dtype=dtype),
            nn.BatchNorm2d(out_channels, dtype=dtype),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
    def forward(self, X):
        
        return self.block(X)
    
class UpBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, dtype=float):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dtype=dtype),
            nn.BatchNorm2d(out_channels, dtype=dtype),
            nn.ReLU())
        
    def forward(self, X):
        
        return self.block(X)

class DomainDistinguisher(nn.Module):
    
    def __init__(self, layers=[3, 16, 32, 64, 128], dtype=float):
        super().__init__()
        
        self.downs = nn.ModuleList()
        for i in range(len(layers)-1):
            self.downs.append(DownBlock(layers[i], layers[i+1]))
            
        self.ups = nn.ModuleList()
        for i in range(len(layers)-1, 0, -1):
            self.ups.append(UpBlock(layers[i], layers[i-1]))
            
    
    def forward(self, X):
        

        for d in self.downs:
            X = d(X)
            
        for u in self.ups:
            X = u(X)
            
        return X
         
            
class MYCNDistinguisher(nn.Module):
    
    def __init__(self, layers=[3, 16, 32, 64, 128], dtype=float, inp_size=128):
        super().__init__()
        
        self.downs = nn.ModuleList()
        for i in range(len(layers)-1):
            self.downs.append(DownBlock(layers[i], layers[i+1]))
          
        w_h = inp_size / 2**(len(layers)-1)
        fc_nodes = int(layers[-1] * w_h ** 2)
                    
        self.FC = nn.Sequential(
            nn.Linear(fc_nodes, 1000, dtype=dtype),
            nn.ReLU(),
            nn.Linear(1000, 100, dtype=dtype),
            nn.ReLU(),
            nn.Linear(100,1, dtype=dtype)
        )
            
    
    def forward(self, X):
        
        for d in self.downs:
            X = d(X)
            
        return self.FC(X.flatten(start_dim=1)).squeeze()
    
    
class MYCNDomainAdaptedDistinguisher(nn.Module):
        
    def __init__(self, layers=[3, 16, 32, 64, 128], dtype=float, inp_size=128, labels=4):
        super().__init__()
        
        self.downs = nn.ModuleList()
        for i in range(len(layers)-1):
            self.downs.append(DownBlock(layers[i], layers[i+1]))
          
        w_h = inp_size / 2**(len(layers)-1)
        fc_nodes = int(layers[-1] * w_h ** 2)
                    
        self.MYCN_FC = nn.Sequential(
            nn.Linear(fc_nodes, 1000, dtype=dtype),
            nn.ReLU(),
            nn.Linear(1000, 100, dtype=dtype),
            nn.ReLU(),
            nn.Linear(100,1, dtype=dtype)
        )
        
        self.Domain_FC = nn.Sequential(
            nn.Linear(fc_nodes, 1000, dtype=dtype),
            nn.ReLU(),
            nn.Linear(1000, 100, dtype=dtype),
            nn.ReLU(),
            nn.Linear(100, labels, dtype=dtype)
        )
            
    
    def forward(self, X):
        
        for d in self.downs:
            X = d(X)
            
        MYCN = self.MYCN_FC(X.flatten(start_dim=1)).squeeze()  
        DOMAIN = self.Domain_FC(X.flatten(start_dim=1)).squeeze() 
        
        return MYCN, torch.sigmoid(DOMAIN)
