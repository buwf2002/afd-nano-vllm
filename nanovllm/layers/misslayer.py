import torch.nn as nn

class MissLayer(nn.Module):
    def __init__(self, *args, **kwargs):  
        super().__init__() 
    
    def forward(self, *args, **kwargs):  
        """Return the first arg from args or the first value from kwargs."""  
        return args[0] if args else next(iter(kwargs.values()))