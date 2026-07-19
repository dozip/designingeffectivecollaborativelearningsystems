import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import numpy as np


class Net(nn.Sequential):
    """Fully connected regression network.

    The network maps ``num_features`` input features through hidden
    layers of sizes 48, 24, 12, and 4 to one scalar output.

    Args:
        num_features:
            Number of features in the last dimension of the input tensor.
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


# Generic PyTorch training and evaluation helpers for regression models.
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
    """Client-side feature extractor for the split-learning architecture.

    The network maps ``num_features`` input features through two
    batch-normalized hidden layers and produces a 24-dimensional
    intermediate representation for the server model.

    Args:
        num_features:
            Number of features in the last dimension of the input tensor.
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
    """Server-side transformation for the split-learning architecture.

    The model receives a 24-dimensional client representation, transforms
    it through batch-normalized hidden layers of sizes 144 and 80, and
    returns another 24-dimensional representation.
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
#     """Linear client-side prediction head.

    # Args:
    #     n_input:
    #         Number of features in the incoming representation.
    #     n_output:
    #         Number of prediction features produced by the linear layer.
    # """


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
    """Linear client-side prediction head.

    Args:
        n_input:
            Number of features in the incoming representation.
        n_output:
            Number of prediction features produced by the linear layer.
    """

    def __init__(self, n_input, n_output) -> None:
        super(NetLocal2, self).__init__()
        self.lin1 = nn.Linear(n_input, n_output)


    def forward(self, x):
        z = self.lin1(x)
        return z
    
class LSTM_Model(nn.Sequential):
    """Single-layer, batch-first LSTM feature extractor.

    The model returns the complete sequence of hidden states. Consequently,
    the output feature dimension equals ``n_hidden``.

    Args:
        n_input:
            Number of features per input time step.
        n_output:
            Retained for compatibility with existing constructor calls.
            This parameter is currently not used by the implementation.
        n_hidden:
            Number of LSTM hidden features and therefore the size of the
            last dimension of the returned tensor.
    """

    def __init__(self, n_input, n_output, n_hidden):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size = n_input, hidden_size = n_hidden, num_layers=1, batch_first = True)

    def forward(self, input):
        x, _ = self.lstm(input)

        return x

class Dense(nn.Sequential):
    """Single linear projection layer.

    Args:
        n_input:
            Number of input features.
        n_output:
            Number of output features.
    """

    def __init__(self, n_input, n_output) -> None:
        super(Dense, self).__init__()
        self.lin1 = nn.Linear(n_input, n_output)


    def forward(self, x):
        z = self.lin1(x)
        return z