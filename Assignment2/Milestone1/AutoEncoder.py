import torch
import torch.nn as nn
import torch.nn.init as init

class Autoencoder(nn.Module):
    def __init__(self, dims):
        super(Autoencoder, self).__init__()

        self.dims = dims

        latent_space_ind = 0
        latent_space_dim = 1e5

        for i in range(len(self.dims)):
            if self.dims[i] < latent_space_dim:
                latent_space_dim = self.dims[i]
                latent_space_ind = i

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(self.dims)-1):
            if i < latent_space_ind:
                self.encoder.append(nn.Linear(dims[i], dims[i+1]))
                self.encoder.append(nn.ReLU())
            else:
                self.decoder.append(nn.Linear(dims[i], dims[i+1]))
                self.decoder.append(nn.ReLU())
            
        self.decoder = self.decoder[:-1] # removing last layer

        # Weight initialization
        self.init_weights()

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def encode(self, x):
        for l in self.encoder:
            x = l(x)
        return x
    
    def decode(self, x):
        for l in self.decoder:
            x = l(x)
        return x

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)