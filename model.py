from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class MVE(nn.Module):
    def __init__(self,c1=5,c2=7,c3=13):
        super(MVE, self).__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=c1,stride=1,padding=c1 // 2)
        self.pool2 = nn.MaxPool1d(kernel_size=c2, stride=1, padding=c2 // 2)
        self.pool3 = nn.MaxPool1d(kernel_size=c3, stride=1, padding=c3 // 2)
    def forward(self,x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        return torch.cat([x,x1,x2,x3],dim=1)

class MVC(nn.Module):
    def __init__(self, c1, c2, k=1,cut=0):
        super(MVC, self).__init__()
        self.cut = cut
        self.conv = nn.Conv1d(c1 * 4, c2, k, 1)
    def forward(self, x): 
        x = x[..., : x.size(dim=-1)-self.cut]
        x = self.conv(torch.cat([x[..., ::4], x[..., 1::4], x[..., 2::4], x[..., 3::4]], 1))
        return x


class MVSleepNet_base(pl.LightningModule):   ###     
    def __init__(self):
        super().__init__()
        self.spp1 = MVE()
        self.spp2 = MVE()
        self.spp3 = MVE()

        
        self.maxpool1 = nn.MaxPool1d(2, 2)
        self.maxpool2 = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(8)

        self.fcus1 = MVC(4,8)
        self.fcus2 = MVC(32,64,cut=2)
        self.fcus3 = MVC(256,512,cut=3)
        self.conv1 = nn.Conv1d(512, 128, 1)
        self.conv2 = nn.Conv1d(128, 32, 1)
        self.conv3 = nn.Conv1d(32, 8, 1)
        self.do0 = nn.Dropout()
        self.lstm = nn.GRU(input_size=368, hidden_size=128, batch_first=True)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):                       #[1,20,1,3000]
        old_shape = x.shape                     
        x = x.chunk(x.shape[0], 0)
        x = torch.cat(x, 1).squeeze(0)          #[20,1,3000]
        x = self.spp1(x)                        #[20,4,3000]
        x = F.relu(self.bn1(self.fcus1(x)))     #[20,8,750]
        x = self.spp2(x)                        #[20,32,750]
        x = F.relu(self.bn2(self.fcus2(x)))     #[20,64,187]
        x = self.spp3(x)                        #[20,256,187]
        x = F.relu(self.bn3(self.fcus3(x)))     #[20,512,46]
        x = F.relu(self.bn4(self.conv1(x)))     #[20,128,46]
        x = F.relu(self.bn5(self.conv2(x)))     #[20,32,46]
        x = F.relu(self.bn6(self.conv3(x)))     #[20,8,46]
        x = self.flatten(x)                     #[20,368]
        x = x.unsqueeze(0).chunk(old_shape[0], 1)
        x = torch.cat(x)
        x = self.do0(x)
        # x = self.fc0(x)
        x, _ = self.lstm(x)
        x = self.fc2(x)
        return(x)


class MVSleepNet_tiny(pl.LightningModule):   ###     
    def __init__(self):
        super().__init__()
        self.spp1 = MVE()
        self.spp2 = MVE()
        
        self.maxpool1 = nn.MaxPool1d(2, 2)
        self.maxpool2 = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(8)

        self.fcus1 = MVC(4,8)
        self.fcus2 = MVC(32,64,cut=2)
        self.fcus3 = MVC(64,128,cut=3)
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.conv2 = nn.Conv1d(64, 8, 1)
        self.do0 = nn.Dropout()
        self.lstm = nn.GRU(input_size=368, hidden_size=128, batch_first=True)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):                       #[1,20,1,3000]
        old_shape = x.shape                     
        x = x.chunk(x.shape[0], 0)
        x = torch.cat(x, 1).squeeze(0)          #[20,1,3000]
        x = self.spp1(x)                        #[20,4,3000]
        x = F.relu(self.bn1(self.fcus1(x)))     #[20,8,750]
        x = self.spp2(x)                        #[20,32,750]
        x = F.relu(self.bn2(self.fcus2(x)))     #[20,64,187]
        x = F.relu(self.bn3(self.fcus3(x)))     #[20,128,46]
        x = F.relu(self.bn4(self.conv1(x)))     #[20,64,46]
        x = F.relu(self.bn5(self.conv2(x)))     #[20,8,46]
        x = self.flatten(x)                     #[20,368]
        x = x.unsqueeze(0).chunk(old_shape[0], 1)
        x = torch.cat(x)
        x = self.do0(x)
        # x = self.fc0(x)
        x, _ = self.lstm(x)
        x = self.fc2(x)
        return(x)



