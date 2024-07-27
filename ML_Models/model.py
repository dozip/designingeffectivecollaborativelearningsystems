import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import numpy as np


class Net(nn.Sequential):
    """_summary_

    Args:
        nn (_type_): parent class
    """

    def __init__(self, num_features) -> None:
        super(Net, self).__init__()
        self.lin1 = nn.Linear(num_features, 48)
        self.lin2 = nn.Linear(48, 24)
        self.lin3 = nn.Linear(24, 12)
        self.lin4 = nn.Linear(12, 4)
        self.lin5 = nn.Linear(4, 1)

    def forward(self, x):
        z = torch.relu(self.lin1(x))
        z = torch.relu(self.lin2(z))
        z = torch.relu(self.lin3(z))
        z = torch.relu(self.lin4(z))
        z = self.lin5(z)
        return z


## for federated learning
def train(net: Net, trainloader, optimizer, epochs, deivce: str):
    criterion = torch.nn.MSELoss()
    net.train()
    net.to(device=deivce)

    for _ in range(epochs):

        for images, labels in trainloader:
            images, labels = images.to(deivce), labels.to(deivce)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()


def test(net: Net, testloader: DataLoader, device: str):
    criterion = torch.nn.MSELoss()
    correct, loss = 0, 0.0
    accuracy = 0

    net.eval()
    net.to(device)

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()

    return loss, loss

### Models for Split Learning
class NetLocal1(nn.Sequential):
    """_summary_

    Args:
        nn (_type_): parent class
    """

    def __init__(self, num_features:int) -> None:
        super(NetLocal1, self).__init__()
        self.lin1 = nn.Linear(num_features, 128)
        self.batch1_norm_1d = nn.BatchNorm1d(128)
        self.lin2 = nn.Linear(128, 52)
        self.batch2_norm_1d = nn.BatchNorm1d(52)
        self.lin3 = nn.Linear(52, 24)

    def forward(self, x):
        z = torch.relu(self.batch1_norm_1d(self.lin1(x)))
        z = torch.relu(self.batch2_norm_1d(self.lin2(z)))
        z = self.lin3(z)
        return z
    
class NetServerModel(nn.Sequential):
    """_summary_

    Args:
        nn (_type_): parent class
    """

    def __init__(self) -> None:
        super(NetServerModel, self).__init__()
        self.lin1 = nn.Linear(24, 144)
        self.batch1_norm_1d = nn.BatchNorm1d(144)
        self.lin2 = nn.Linear(144, 80)
        self.batch2_norm_1d = nn.BatchNorm1d(80)
        self.lin3 = nn.Linear(80, 24)

    def forward(self, x):
        z = torch.relu(self.batch1_norm_1d(self.lin1(x)))
        z = torch.relu(self.batch2_norm_1d(self.lin2(z)))
        z = self.lin3(z)
        return z

# class NetLocal2(nn.Sequential):
#     """_summary_

#     Args:
#         nn (_type_): parent class
#     """


#     def __init__(self, num_target: int) -> None:
#         super(NetLocal2, self).__init__()
#         self.lin1 = nn.Linear(24, 48)
#         self.batch1_norm_1d = nn.BatchNorm1d(48)
#         self.lstm = nn.LSTM(input_size = 48, hidden_size = 48, num_layers=1)
#         self.lin2 = nn.Linear(48, 24)
#         self.batch2_norm_1d = nn.BatchNorm1d(24)
#         self.lin3 = nn.Linear(24, 12)
#         self.batch3_norm_1d = nn.BatchNorm1d(12)
#         self.lin4 = nn.Linear(12, num_target*2)
#         self.batch4_norm_1d = nn.BatchNorm1d(4)
#         self.lin5 = nn.Linear(num_target*2, num_target)


#     def forward(self, x):
#         z = torch.relu(self.batch1_norm_1d(self.lin1(x)))
#         z,_ = self.lstm(z)
#         z = torch.relu(self.batch2_norm_1d(self.lin2(z)))
#         z = torch.relu(self.batch3_norm_1d(self.lin3(z)))
#         z = torch.relu(self.batch4_norm_1d(self.lin4(z)))
#         z = self.lin5(z)
#         return z

class NetLocal2(nn.Sequential):
    """_summary_

    Args:
        nn (_type_): parent class
    """

    def __init__(self, n_input, n_output) -> None:
        super(NetLocal2, self).__init__()
        self.lin1 = nn.Linear(n_input, n_output)


    def forward(self, x):
        z = self.lin1(x)
        return z
    
class LSTM_Model(nn.Sequential):

    def __init__(self, n_input, n_output, n_hidden):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size = n_input, hidden_size = n_hidden, num_layers=1, batch_first = True)

    def forward(self, input):
        x, _ = self.lstm(input)

        return x

class Dense(nn.Sequential):
    """_summary_

    Args:
        nn (_type_): parent class
    """

    def __init__(self, n_input, n_output) -> None:
        super(Dense, self).__init__()
        self.lin1 = nn.Linear(n_input, n_output)


    def forward(self, x):
        z = self.lin1(x)
        return z