import logging
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ML_Models.model import LSTM_Model, NetLocal2
from helpers.helper_classes import MultiChannel_LSTM, EarlyStopping
from helpers.helpers import create_dataset, select_gpu

from .base import ForecastingBackend
from . import register_backend

logger = logging.getLogger('logger')

def _cfg_get_early_stopping(cfg, key, default):
    """Read one global early-stopping setting from YAML.

    Preferred YAML section for all trainable backends:
        early_stopping:
          patience: 100
          min_delta: 0.0

    The default keeps the old behavior if the key is not configured.
    """
    es_cfg = cfg.get("early_stopping", {}) if isinstance(cfg, dict) else {}
    if isinstance(es_cfg, dict) and key in es_cfg:
        return es_cfg[key]
    return default


def _snapshot_local_lstm(lstm_models, dense_models, scaler):
    """Store only trainable state for early stopping, not optimizers or live objects."""
    return {
        "lstm_model": [deepcopy(model.state_dict()) for model in lstm_models],
        "dense_model": [deepcopy(model.state_dict()) for model in dense_models],
        "scaler": deepcopy(scaler),
    }


def _restore_local_lstm_snapshot(snapshot, lstm_models, dense_models):
    """Restore the selected best state into the existing model objects."""
    for model, state in zip(lstm_models, snapshot["lstm_model"]):
        model.load_state_dict(state)

    for model, state in zip(dense_models, snapshot["dense_model"]):
        model.load_state_dict(state)

    return snapshot["scaler"]


@register_backend("local_multichannel")
class LSTMLocalBackend(ForecastingBackend):
    """Per-agent multi-channel LSTM, trained locally for each agent.

    Inference uses the vanilla per-agent loop: each agent's trained
    ``MultiChannel_LSTM`` is attached and queried through ``agent.act``.
    """

    name = "local_multichannel"

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[list[float]]:
        return _local_training_multichannel_lstm(
            simulation, market, supply_chain, sc_agent_list, self.cfg
        )


def _local_training_multichannel_lstm(simulation, market, supply_chain, sc_agent_list, cfg):

    loss_cal = "aggregated"  # individual or aggregated
    loss_fn = nn.L1Loss()
    device = select_gpu()
    pin_memory = getattr(device, "type", None) == "cuda"
    val_loss_list = []

    for i, agent in enumerate(sc_agent_list[1]):  # change 1 to variable for dynamics
        logger.info(f"Training agent: {i}")

        val_loss_agent = []

        lstm_input_dim = 1
        lstm_output_dim = 24
        lstm_hidden_dim = 48
        dense_input_dim = agent.num_retailer * lstm_hidden_dim
        dense_ouput_dim = 1

        learning_rate = 0.001
        momentum = 0.9

        train_size = simulation.train_size
        vall_size = simulation.val_size
        patience = int(_cfg_get_early_stopping(cfg, "patience", 400))
        min_delta = float(_cfg_get_early_stopping(cfg, "min_delta", 0.0))
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            delta=min_delta,
        )

        # 1) Create Model
        lstm_models = []
        dense_models = []
        lstm_optimizers = []
        dense_optimizers = []
        for j in range(agent.num_retailer):
            lstm = LSTM_Model(n_input=lstm_input_dim, n_output=lstm_output_dim, n_hidden=lstm_hidden_dim).to(device)
            lstm_optim = torch.optim.SGD(lstm.parameters(), lr=learning_rate, momentum=momentum)
            lstm_models.append(lstm)
            lstm_optimizers.append(lstm_optim)
            dense = NetLocal2(n_input=dense_input_dim, n_output=dense_ouput_dim).to(device)
            dense_optim = torch.optim.SGD(dense.parameters(), lr=learning_rate, momentum=momentum)
            dense_models.append(dense)
            dense_optimizers.append(dense_optim)
        # 2) Create Data
        trainloaders = []
        val_data_list = []

        # build dataset and dataloader for training of agents defined
        raw_demand_data = agent.demand_by_retailer_history
        logger.debug("Raw demand data length: %s", len(raw_demand_data))

        data = []
        logger.debug("Num Retailer: %s", agent.num_retailer)
        for j in range(agent.num_retailer):
            data.append(np.array(raw_demand_data[j]))

        df = pd.DataFrame(data).transpose()
        df_train = df.iloc[-(train_size + vall_size):-(vall_size), :]
        df_val = df.iloc[-(vall_size):, :]
        scaler = StandardScaler()
        scaler = scaler.fit(df_train.to_numpy())
        data_train_scaled = scaler.transform(df_train)
        data_val_scaled = scaler.transform(df_val)

        # training data
        agent_val_data = []
        for j in range(agent.num_retailer):
            # get data for retailer
            train_data = data_train_scaled[:, j].reshape(train_size, 1)
            val_data = data_val_scaled[:, j].reshape(vall_size, 1)
            X_train, y_train = create_dataset(train_data, lookback=agent.sequence_length)
            X_val, y_val = create_dataset(val_data, lookback=agent.sequence_length)
            dataloader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=agent.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
            )
            trainloaders.append(dataloader)
            val_data_list.append([X_val, y_val])

        datasets = [None] * len(trainloaders)

        # 3) Train

        for epoch in range(agent.epochs):
            logger.debug(f"Number Epoch: {epoch}")
            training_loss_epoch = [0]

            # go over every batch
            for batches in zip(*trainloaders):
                logger.debug("New Batch")
                training_loss_epoch = [0] * (agent.num_retailer)

                for model in lstm_models:
                    model.train()
                for optim in lstm_optimizers:
                    optim.zero_grad()
                for model in dense_models:
                    model.train()
                for optim in dense_optimizers:
                    optim.zero_grad()

                # get data of batch
                features_list = []
                label_list = []

                for j, batch in enumerate(batches):
                    features_list.append(batch[0].to(device, non_blocking=pin_memory))
                    label_list.append(batch[1].to(device, non_blocking=pin_memory))

                ######################
                ##### FORWARD ########
                ######################

                # get lstm output
                logger.debug("Output LSTM")
                lstm_outputs = []
                lstm_outputs_detached = []

                for j, feature in enumerate(features_list):
                    output = lstm_models[j](feature)
                    output_detached = output.clone().detach().requires_grad_(True)
                    lstm_outputs.append(output)
                    lstm_outputs_detached.append(output_detached)

                logger.debug("Feature Fusion")
                fusion = torch.cat(lstm_outputs_detached, axis=2)
                # get dense output
                logger.debug("Output Dense")
                dense_outputs = []
                dense_outputs_detached = []

                for j, idx in enumerate(features_list):
                    output = dense_models[j](fusion)
                    output_detached = output.clone().detach().requires_grad_(True)
                    dense_outputs.append(output)
                    dense_outputs_detached.append(output_detached)

                # individual loss
                loss_list = []
                loss_list_item = []
                if loss_cal == "individual":
                    logger.debug("Loss individual")
                    loss_list = []
                    loss_list_item = []
                    for i, output in enumerate(dense_outputs):
                        loss = loss_fn(output, label_list[i])
                        loss_list.append(loss)
                        loss_list_item.append(loss.item())

                if loss_cal == "aggregated":
                    logger.debug("Loss aggregated")
                    fusion_outputs = torch.cat(dense_outputs, axis=2)
                    fusion_targets = torch.cat(label_list, axis=2)
                    loss_list = [loss_fn(fusion_outputs, fusion_targets)] * (agent.num_retailer)
                    loss_list_item = [loss_fn(fusion_outputs, fusion_targets).item()] * (agent.num_retailer)

                training_loss_epoch = [sum(x) for x in zip(training_loss_epoch, loss_list_item)]
                ######################
                ##### BACKWARD #######
                ######################

                ### backward for dense
                gradients = []
                for j, output in enumerate(dense_outputs):
                    if loss_cal == "aggregated":
                        if j == 0:
                            loss_list[j].backward()
                    else:
                        loss_list[j].backward()
                    gradients.append(lstm_outputs_detached[j].grad)

                # backward for lstm
                for j, output in enumerate(lstm_outputs):
                    output.backward(gradients[j])

                ######################
                ##### Optim Step #####
                ######################
                for optimizer in lstm_optimizers:
                    optimizer.step()

                for optimizer in dense_optimizers:
                    optimizer.step()
            ######################
            ##### VALIDATION #####
            ######################

            logger.debug("Validation")
            # set models to eval mode mode
            for model in lstm_models:
                model.eval()
            for model in dense_models:
                model.eval()
            with torch.no_grad():

                # get lstm output
                logger.debug("Output LSTM")
                label_list = []
                rescaled_label_list = []
                lstm_outputs = []
                lstm_outputs_detached = []

                for i, idx in enumerate(range(agent.num_retailer)):
                    output = lstm_models[i](val_data_list[i][0].to(device, non_blocking=pin_memory))
                    output_detached = output.clone().detach().requires_grad_(True)
                    lstm_outputs.append(output)
                    lstm_outputs_detached.append(output_detached)

                    label_list.append(val_data_list[i][1][:, -1, :])  # use only last time step as target for validation loss

                    rescaled_labels = val_data_list[i][1][:, -1, :] * scaler.scale_[i] + scaler.mean_[i]
                    rescaled_label_list.append(rescaled_labels)

                # server routine
                logger.debug("Feature Fusion")
                fusion = torch.cat(lstm_outputs_detached, axis=2)

                # get dense output
                logger.debug("Output Dense")
                dense_outputs = []
                dense_outputs_detached = []
                for i, idx in enumerate(range(agent.num_retailer)):
                    output = dense_models[i](fusion)
                    output_detached = output.clone().detach().requires_grad_(True)
                    dense_outputs.append(output)
                    dense_outputs_detached.append(output_detached)

                # loss calcualtion
                loss_list = []
                loss_list_item = []
                if loss_cal == "individual":
                    logger.debug("Loss individual")
                    loss_list = []
                    loss_list_item = []
                    for i, output in enumerate(dense_outputs):
                        loss = loss_fn(output[:, -1, :].cpu() * scaler.scale_[i] + scaler.mean_[i], rescaled_label_list[i])
                        loss_list.append(loss)
                        loss_list_item.append(loss.item())

                if loss_cal == "aggregated":
                    logger.debug("Loss aggregated")
                    outputs = []
                    targets = []
                    for i, output in enumerate(dense_outputs):
                        outputs.append(dense_outputs[i][:, -1, :].cpu() * scaler.scale_[i] + scaler.mean_[i])  # use only last time step of predicted values for validation loss
                        targets.append(rescaled_label_list[i])

                    fusion_outputs = torch.cat(outputs, axis=1)
                    fusion_targets = torch.cat(targets, axis=1)
                    loss = loss_fn(fusion_outputs, fusion_targets)
                    loss_list.append(loss)
                    loss_list_item.append(loss.item())

                sum_val_loss = np.sum(loss_list_item)
                # validation_loss.append(sum_val_loss)

                logger.debug(f"validation_loss_sum: {sum_val_loss}")
                logger.debug(f"validation_loss_per_agent: {loss_list_item}")

                # ceck for early stopping
                snapshot = _snapshot_local_lstm(lstm_models, dense_models, scaler)
                early_stopping(sum_val_loss, snapshot)

                if early_stopping.early_stop:
                    logger.debug("Early stopping")
                    epoch_diff = agent.epochs - epoch
                    #### if early stopping is reached, then use the last loss values for the rest of the planned epochs
                    for k in range(epoch_diff):
                        val_loss_agent.append(sum_val_loss)
                    break
                val_loss_agent.append(sum_val_loss)

        best_snapshot = early_stopping.best_model
        if best_snapshot is not None:
            scaler = _restore_local_lstm_snapshot(
                best_snapshot,
                lstm_models,
                dense_models,
            )
        logger.debug(f"Training-Loss: {training_loss_epoch}")

        model = MultiChannel_LSTM(num_channels=agent.num_retailer, lstm_model=lstm_models, dense_model=dense_models, scaler=scaler, device=device)

        agent.set_forecasting_model(model)
        val_loss_list.append(val_loss_agent)

    # Convert to a NumPy array
    logger.debug(f"Number of validation loss series: {len(val_loss_list)}")
    logger.debug(f"Validation loss lengths per agent: {[len(losses) for losses in val_loss_list]}")
    val_loss = np.array(val_loss_list)

    # Sum over columns
    val_loss = np.sum(val_loss, axis=0)

    return val_loss
