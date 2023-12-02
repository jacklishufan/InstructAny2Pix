from torch import nn

class DummyLoss(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()

