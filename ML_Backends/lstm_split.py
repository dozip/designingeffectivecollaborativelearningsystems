import logging
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


@register_backend("split_multichannel")
class LSTMSplitBackend(ForecastingBackend):
    """Collaborative split-learning multi-channel LSTM.

    Training is collaborative across the level-1 agents. Inference is also
    collaborative: at level 1 the LSTM outputs of every agent are fused
    before each agent's dense head produces its forecast, so the predictions
    must be computed jointly via ``collaborative_predict``.
    """

    name = "split_multichannel"

    def train(self, simulation, market, supply_chain, sc_agent_list) -> Optional[list[float]]:
        return _split_training_multichannel_lstm(
            simulation, market, supply_chain, sc_agent_list, self.cfg
        )

    @property
    def collaborative_level(self) -> Optional[int]:
        return 1

    def collaborative_predict(self, level_agents, demand_t, sc_agent_list, t) -> list:
        device = None
        for agent in level_agents:
            agent_lstm = agent.get_forecasting_model().lstm_model
            if agent_lstm:
                try:
                    device = next(agent_lstm[0].parameters()).device
                except StopIteration:
                    device = None
                break
        if device is None:
            device = select_gpu()

        # 1) build features
        datasets = []
        for k, agent in enumerate(level_agents):
            raw_demand_data = agent.demand_by_retailer_history

            data = []
            for l in range(agent.num_retailer):
                data.append(np.array(raw_demand_data[l]))

            df = pd.DataFrame(data).transpose()
            df = df.iloc[-(agent.sequence_length - 1):, :]
            df.loc[len(df)] = demand_t[:, k]
            data_scaled = agent.forecasting_model.scaler.transform(df.to_numpy())

            for l in range(agent.num_retailer):
                train_data = data_scaled[-agent.sequence_length:, l].reshape(agent.sequence_length, 1)
                datasets.append(train_data)

        # 2) inference
        # 2.1) get output lstm
        output_lstm_list = []
        retailer_offset = 0
        for k, agent in enumerate(level_agents):
            data = np.array(datasets[retailer_offset:(retailer_offset + agent.num_retailer)])
            retailer_offset += agent.num_retailer

            agent_lstm = agent.get_forecasting_model().lstm_model

            for l, lstm in enumerate(agent_lstm):
                lstm.eval()
                input_tensor = torch.tensor([np.array(data[l])], dtype=torch.float32).to(device)
                output_lstm_list.append(lstm(input_tensor))

        # 2.2) Feature Fusion
        fusion = torch.cat((output_lstm_list), axis=2)

        # 2.3) get dense output
        output_dense_list = []
        for k, agent in enumerate(level_agents):
            agent_forecastingmodel = agent.get_forecasting_model()
            agent_dense = agent_forecastingmodel.dense_model
            for l, dense in enumerate(agent_dense):
                dense.eval()
                all_output = dense(fusion)
                output = all_output[:, -1, :]
                output_rescaled = (output.item() * agent_forecastingmodel.scaler.scale_[l]) + agent_forecastingmodel.scaler.mean_[l]

                output_dense_list.append(output_rescaled)

        # 3) split prediction across agents
        predictions = []
        for k, agent in enumerate(level_agents):
            predictions.append(output_dense_list[(k * agent.num_retailer):((k + 1) * agent.num_retailer)])

        return predictions


def _split_training_multichannel_lstm(simulation, market, supply_chain, sc_agent_list, cfg):
    logger.info("Starting Syncron Training")
    loss_cal = "aggregated"  # individual or aggregated
    loss_fn = nn.L1Loss()
    device = select_gpu()

    agent_models = {}

    lstm_input_dim = 1
    lstm_output_dim = 24
    lstm_hidden_dim = 48

    dense_ouput_dim = 1

    learning_rate = 0.001
    momentum = 0.9

    epochs = cfg['sim']['epochs']
    num_supplier = len(sc_agent_list[1])

    train_size = simulation.train_size
    vall_size = simulation.val_size
    patience = 400
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # 1) Create Models
    for i, agent in enumerate(sc_agent_list[1]):
        agent_key = "agent_" + str(i)
        agent_models[agent_key] = {}
        lstm_models = []
        dense_models = []
        lstm_optimizers = []
        dense_optimizers = []
        dataloaders_id = []
        for j in range(agent.num_retailer):
            dataloader_id = i * agent.num_retailer + j
            dataloaders_id.append(dataloader_id)
            lstm = LSTM_Model(n_input=lstm_input_dim, n_output=lstm_output_dim, n_hidden=lstm_hidden_dim)
            lstm.to(device)
            lstm_optim = torch.optim.SGD(lstm.parameters(), lr=learning_rate, momentum=momentum)
            lstm_models.append(lstm)
            lstm_optimizers.append(lstm_optim)
            dense_input_dim = (agent.num_retailer * num_supplier) * lstm_hidden_dim
            dense = NetLocal2(n_input=dense_input_dim, n_output=dense_ouput_dim)
            dense.to(device)
            dense_optim = torch.optim.SGD(dense.parameters(), lr=learning_rate, momentum=momentum)
            dense_models.append(dense)
            dense_optimizers.append(dense_optim)

        agent_models[agent_key]['lstm_model'] = lstm_models
        agent_models[agent_key]['lstm_optim'] = lstm_optimizers

        agent_models[agent_key]['dense_model'] = dense_models
        agent_models[agent_key]['dense_optim'] = dense_optimizers
        agent_models[agent_key]['dataloader_ids'] = dataloaders_id
    # 2) Create Data

    trainloaders = []
    val_data_list = []
    scaler_list = []

    for i, agent in enumerate(sc_agent_list[1]):  # change 1 to variable for dynamics
        logger.info(f"start traing multichannel lstm for agent: {i}")

        # build dataset and dataloader for training of agents defined
        raw_demand_data = agent.demand_by_retailer_history

        data = []
        for r in range(agent.num_retailer):
            data.append(np.array(raw_demand_data[r]))

        df = pd.DataFrame(data).transpose()
        df_train = df.iloc[-(train_size + vall_size):-(vall_size), :]
        df_val = df.iloc[-(vall_size):, :]
        scaler = StandardScaler()
        scaler = scaler.fit(df_train.to_numpy())
        scaler_list.append(scaler)
        data_train_scaled = scaler.transform(df_train)
        data_val_scaled = scaler.transform(df_val)

        agent_val_data = []
        for j in range(agent.num_retailer):
            # get data for retailer
            train_data = data_train_scaled[:, j].reshape(train_size, 1)
            val_data = data_val_scaled[:, j].reshape(vall_size, 1)
            X_train, y_train = create_dataset(train_data, lookback=agent.sequence_length)
            X_val, y_val = create_dataset(val_data, lookback=agent.sequence_length)
            dataloader = DataLoader(TensorDataset(X_train, y_train),
                                    batch_size=agent.batch_size, shuffle=False, num_workers=1)
            trainloaders.append(dataloader)
            agent_val_data.append([X_val, y_val])
        val_data_list.append(agent_val_data)

    datasets = [None] * len(trainloaders)
    # 3) Training
    val_loss = []
    # train over every epoch
    for epoch in range(epochs):
        logger.info(f"Number Epoch: {epoch}")
        # go over every batch
        for batches in zip(*trainloaders):
            logger.debug("New Batch")

            if loss_cal == "aggregated":
                training_loss_epoch = [0] * (num_supplier)

            # set models to training mode
            for i, agent in enumerate(sc_agent_list[1]):
                agent_key = "agent_" + str(i)
                for j in range(agent.num_retailer):
                    lstm_models = agent_models[agent_key]['lstm_model']
                    lstm_optimizers = agent_models[agent_key]['lstm_optim']
                    dense_models = agent_models[agent_key]['dense_model']
                    dense_optimizers = agent_models[agent_key]['dense_optim']
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

            for i, batch in enumerate(batches):
                features_list.append(batch[0].to(device))
                label_list.append(batch[1].to(device))

            ######################
            ##### FORWARD ########
            ######################

            # get lstm output
            logger.debug("Output LSTM")
            lstm_outputs = []
            lstm_outputs_detached = []
            for agent_key in agent_models:
                lstm_models = agent_models[agent_key]['lstm_model']
                for i, id in enumerate(agent_models[agent_key]['dataloader_ids']):
                    output = lstm_models[i](features_list[id])
                    output_detached = output.clone().detach().requires_grad_(True)
                    lstm_outputs.append(output)
                    lstm_outputs_detached.append(output_detached)

            # server routine
            logger.debug("Feature Fusion")
            fusion = torch.cat(lstm_outputs_detached, axis=2)

            # get dense output
            logger.debug("Output Dense")
            dense_outputs = []
            dense_outputs_detached = []
            for agent_key in agent_models:
                dense_models = agent_models[agent_key]['dense_model']
                for i, id in enumerate(agent_models[agent_key]['dataloader_ids']):
                    output = dense_models[i](fusion)
                    output_detached = output.clone().detach().requires_grad_(True)
                    dense_outputs.append(output)
                    dense_outputs_detached.append(output_detached)

            # loss calulation
            loss_list = []
            loss_list_item = []
            if loss_cal == "individual":
                logger.debug("Loss individual")
                for i, output in enumerate(dense_outputs):
                    loss = loss_fn(output, label_list[i])
                    loss_list.append(loss)
                    loss_list_item.append(loss.item())

            if loss_cal == "aggregated":
                logger.debug("Loss aggregated")
                for agent in agent_models:
                    outputs = []
                    targets = []
                    for id in agent_models[agent]['dataloader_ids']:
                        outputs.append(dense_outputs[id])
                        targets.append(label_list[id])
                    fusion_outputs = torch.cat(outputs, axis=2)
                    fusion_targets = torch.cat(targets, axis=2)
                    loss = loss_fn(fusion_outputs, fusion_targets)
                    loss_list.append(loss)
                    loss_list_item.append(loss.item())

            training_loss_epoch = [sum(x) for x in zip(training_loss_epoch, loss_list_item)]

            ######################
            ##### BACKWARD #######
            ######################

            ### backward for dense
            for loss in loss_list:
                loss.backward()

            # get gradients of lstm
            gradients = []
            for output in lstm_outputs_detached:
                gradients.append(output.grad)

            # backward for lstm
            for i, output in enumerate(lstm_outputs):
                output.backward(gradients[i])

            ######################
            ##### Optim Step #####
            ######################

            for agent_key in agent_models:
                lstm_optim = agent_models[agent_key]['lstm_optim']
                for optimizer in lstm_optim:
                    optimizer.step()

                dense_optim = agent_models[agent_key]['dense_optim']
                for optimizer in dense_optim:
                    optimizer.step()

        logger.info(f"The Training-Loss was: {training_loss_epoch}")

        ######################
        ##### Validation #####
        ######################

        # set models to eval mode mode
        for i, agent in enumerate(sc_agent_list[1]):
            agent_key = "agent_" + str(i)
            for j in range(agent.num_retailer):
                lstm_models = agent_models[agent_key]['lstm_model']
                lstm_optimizers = agent_models[agent_key]['lstm_optim']
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
            for j, agent_key in enumerate(agent_models):
                lstm_models = agent_models[agent_key]['lstm_model']
                for i, id in enumerate(agent_models[agent_key]['dataloader_ids']):
                    output = lstm_models[i](val_data_list[j][i][0].to(device))
                    label_list.append(val_data_list[j][i][1][:, -1, :])  # use only last time step as target for validation loss

                    rescaled_labels = val_data_list[j][i][1][:, -1, :] * scaler_list[j].scale_[i] + scaler_list[j].scale_[i]
                    rescaled_label_list.append(rescaled_labels)

                    output_detached = output.clone().detach().requires_grad_(True)
                    lstm_outputs.append(output)
                    lstm_outputs_detached.append(output_detached)

            # server routine
            logger.debug("Feature Fusion")
            fusion = torch.cat(lstm_outputs_detached, axis=2)

            # get dense output
            logger.debug("Output Dense")
            dense_outputs = []
            dense_outputs_detached = []
            for agent_key in agent_models:
                dense_models = agent_models[agent_key]['dense_model']
                for i, id in enumerate(agent_models[agent_key]['dataloader_ids']):
                    output = dense_models[i](fusion)
                    output_detached = output.clone().detach().requires_grad_(True)
                    dense_outputs.append(output)
                    dense_outputs_detached.append(output_detached)

            # loss calulation
            loss_list = []
            loss_list_item = []
            if loss_cal == "individual":
                raise NotImplementedError()

            if loss_cal == "aggregated":
                logger.debug("Loss aggregated")
                for j, agent in enumerate(agent_models):
                    outputs = []
                    targets = []
                    for i, id in enumerate(agent_models[agent]['dataloader_ids']):
                        outputs.append(dense_outputs[id][:, -1, :].cpu() * scaler_list[j].scale_[i] + scaler_list[j].scale_[i])  # use only last time step of predicted values for validation loss
                        targets.append(rescaled_label_list[id])
                    fusion_outputs = torch.cat(outputs, axis=1)
                    fusion_targets = torch.cat(targets, axis=1)
                    loss = loss_fn(fusion_outputs, fusion_targets)
                    loss_list.append(loss)
                    loss_list_item.append(loss.item())

            sum_val_loss = np.sum(loss_list_item)
            val_loss.append(sum_val_loss)

            logger.info(f"validation_loss_sum: {sum_val_loss}")
            logger.info(f"validation_loss_per_agent: {loss_list_item}")

            # ceck for early stopping
            early_stopping(sum_val_loss, agent_models)

            if early_stopping.early_stop:
                print("Early stopping")
                break
    best_models = early_stopping.best_model

    for j, agent_key in enumerate(best_models):
        lstm_models = best_models[agent_key]['lstm_model']
        dense_models = best_models[agent_key]['dense_model']

        model = MultiChannel_LSTM(num_channels=sc_agent_list[1][j].num_retailer, lstm_model=lstm_models,
                                  dense_model=dense_models, scaler=scaler_list[j], device=device)

        sc_agent_list[1][j].set_forecasting_model(model)

    return val_loss
