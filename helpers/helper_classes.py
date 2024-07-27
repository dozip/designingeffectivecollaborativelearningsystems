import torch
import logging

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
import torch.nn as nn
from ML_Models.model import *
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler


### Def Logger
logger = logging.getLogger('example_logger')


class Replenishment():

    def __init__(self) -> None:
        pass

    def compute_order_sum(self) -> int:
        pass


# class for OrderUpTo Replensihment
class OrderUpTo(Replenishment):

    def __init__(self, R, lead_time, risk_factor, inv_cap) -> None:
        self.R = R
        self.lead_time = lead_time
        self.risk_factor = risk_factor
        self.inv_cap = inv_cap

    def compute_order_sum(self, d_est: int, currnet_inv: int) -> int:
        """compute the complete order size based on current inventory, demand forecast and strategy parameters

        Args:
            d_est (int): _description_
            currnet_inv (int): _description_

        Returns:
            int: _description_
        """

        inv_adjustment = np.round(((self.lead_time + self.risk_factor) * d_est) - currnet_inv)
        order_size = np.round(self.R * d_est + inv_adjustment)

        if order_size < 0:
            order_size = 0
        max_order_size = self.inv_cap - currnet_inv
        if order_size > max_order_size:
            order_size = max_order_size

        return order_size


class Forecasting():

    def __init__(self) -> None:
        pass

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


### Moving Avaerga
class MA(Forecasting):
    """Use Moving Average for demand forecasting

    Args:
        Forecasting (_type_): parent class
    """

    def __init__(self, t: int = 100) -> None:
        super().__init__()
        self.model = None
        self.model_type = "MA"
        self.t = t

    # training is not recessary
    def train(self, data: np.ndarray) -> None:
        pass

    def predict(self, data: np.array) -> int:
        """Computes the average of the input data

        Args:
            demand_data (np.array): demand data

        Returns:
            int: _description_
        """

        d_est = []
        for d in data:
            d_est.append(np.round(np.mean(d[:self.t])))

        return d_est

class SplittNN(Forecasting):

    def __init__(self) -> None:
        super().__init__()
        self.net_1 = None
        self.server_net = None
        self.net_2 = None
        self.scaler = None
    
    def train(self):
        pass

    def test(self):
        pass

    def predict(self, data):

        # set to no training an only inference

        data = (data - self.scaler.mean_) / np.sqrt(self.scaler.var_)
        data = torch.from_numpy(np.array([data]).astype('float32'))
        self.net_1.eval()
        self.server_net.eval()
        self.net_2.eval()

        output_1 = self.net_1.forward(data)
        server_output = self.server_net(output_1)
        demand = self.net_2.forward(server_output)
        output = np.round(demand.detach().cpu().numpy(), 0)[0]
        output = np.sum(output)

        return output
    
    def set_models(self, local_1: nn.Sequential, server: nn.Sequential, local_2: nn.Sequential)-> None:
        self.net_1 = local_1
        self.server_net = server
        self.net_2 = local_2
    
    def set_scaler(self, scaler: StandardScaler) -> None:

        self.scaler = scaler

class MultiChannel_LSTM(Forecasting):

    def __init__(self, num_channels, lstm_model, dense_model, scaler, device) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.lstm_model = lstm_model
        self.dense_model = dense_model
        self.scaler = scaler
        self.device = device

    def train(self):
        pass

    def predict(self, data):
        df = pd.DataFrame(data).transpose()
        data_scaled = self.scaler.transform(df.to_numpy())

        output_lstm = []

        # forward LSTM
        for i in range(self.num_channels):
            self.lstm_model[i].eval()
            input_tensor = torch.tensor([np.array(data_scaled[:,i]).reshape(-1,1)], dtype=torch.float32).to(self.device)
            output = self.lstm_model[i](input_tensor)
            output_lstm.append(output)

        # feature fuesion
        fusion = torch.cat((output_lstm), axis = 2)

        # forward dense
        output_dense = []
        for i in range(self.num_channels):
            output= self.dense_model[i](fusion)[:, -1, :]
            output_rescaled = (output.item()*self.scaler.scale_[i])+self.scaler.mean_[i]
            output_dense.append(output_rescaled)

        return output_dense

class Split_MultiChannel_LSTM(Forecasting):

    def __init__(self, num_channels, lstm_model, dense_model, scaler) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.lstm_model = lstm_model
        self.dense_model = dense_model
        self.scaler = scaler

    def train(self):
        pass

    def predict(self, data):
        df = pd.DataFrame(data).transpose()
        data_scaled = self.scaler.transform(df.to_numpy())

        output_lstm = []

        # forward LSTM
        for i in range(self.num_channels):
            self.lstm_model[i].eval()
            input_tensor = torch.tensor([np.array(data_scaled[:,i]).reshape(-1,1)], dtype=torch.float32)
            output = self.lstm_model[i](input_tensor)
            output_lstm.append(output)

        # feature fuesion
        fusion = torch.cat((output_lstm), axis = 2)

        # forward dense
        output_dense = []
        for i in range(self.num_channels):
            output= self.dense_model[i](fusion)[:, -1, :]
            output_rescaled = (output.item()*self.scaler.scale_[i])+self.scaler.mean_[i]
            output_dense.append(output_rescaled)

        return output_dense

class RegressionDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X.astype('float32'))
        self.y = torch.from_numpy(y.astype('float32'))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # return self.X[index], self.y[index].unsqueeze(0)
        return self.X[index], self.y[index]

    def get_length(self):
        return self.__len__()

    def get_shape(self):

        return self.X.size()

class DataCollector():
    """Create DataSet by adding features and targets for full and complete data sample space
    """

    def __init__(self) -> None:
        self.data = []
        self.target = []
        self.feature_names: list = []
        self.target_names: list = []

        self.num_features = 0
        self.num_targets = 0

    def add_feature(self, feature: dict[str, np.ndarray]):
        for key in feature.keys():
            self.num_features += 1
            self.feature_names.append(key)
            self.data.append(feature[key])

    def add_target(self, target: dict[str, np.ndarray]):
        for key in target.keys():
            self.num_targets += 1
            self.target_names.append(key)
            self.target.append(target[key])

    def return_data(self):
        scaler = StandardScaler()
        data_arr = np.array(self.data)
        data = data_arr.transpose()

        data_transform = scaler.fit_transform(data)

        self.mean = scaler.mean_
        self.var = scaler.var_

        return data_transform, np.array(self.target).transpose(), scaler

    def return_data_as_data_fram(self):
        raise NotImplementedError

    def return_data_loader(self, batch_size: int = 64) -> DataLoader:

        X, y, scaler = self.return_data()
        sample = RegressionDataset(X, y)

        return DataLoader(sample, batch_size=batch_size, shuffle=False, num_workers=0), scaler

    def return_regression_dataset(self) -> RegressionDataset:

        X, y = self.return_data()
        sample = RegressionDataset(X, y)

        return sample
    
    def return_raw_data(self):

        data_arr = np.array(self.data)
        data = data_arr.transpose()

        return data
    
    def return_raw_targets(self):
        target_arr = np.array(self.target)
        print(target_arr.shape)
        target = target_arr.transpose()

        return target

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model = model
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model
            self.counter = 0

    # def __call__(self, val_loss):

    #     score = -val_loss

    #     if self.best_score is None:
    #         self.best_score = score
    #         # self.save_checkpoint(val_loss, model)
    #     elif score < self.best_score + self.delta:
    #         self.counter += 1
    #         self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
    #         if self.counter >= self.patience:
    #             self.early_stop = True
    #     else:
    #         self.best_score = score
    #         # self.save_checkpoint(val_loss, model)
    #         self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss