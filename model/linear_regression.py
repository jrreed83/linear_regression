import torch.nn as nn 

class LinearRegression(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.layer = nn.Linear(din,1) 
    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs