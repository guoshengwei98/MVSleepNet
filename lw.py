import torch.optim
from torch import nn
import pytorch_lightning as pl

from model import MVSleepNet_tiny,MVSleepNet_base

net_base = MVSleepNet_base()
net_tiny = MVSleepNet_tiny()

class LightningWrapper_tiny(pl.LightningModule):
    def __init__(self, net=net_tiny, learning_rate=5e-4):
        super().__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 1.5, 1., 1., 1.]))
        self.learning_rate = learning_rate
        self.max_acc = 0.0
        self.best_k = None
        self.best_f1 = None

    def forward(self, x): 
        x = self.net(x)
        return x


class LightningWrapper_base(pl.LightningModule):
    def __init__(self, net=net_base, learning_rate=1e-4):
        super().__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 1.5, 1., 1., 1.]))
        self.learning_rate = learning_rate
        self.max_acc = 0.0
        self.best_k = None
        self.best_f1 = None

    def forward(self, x):                       #[1,20,1,3000]
        x = self.net(x)
        return x
