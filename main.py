import sys
import os
import glob
import subprocess
import re
import warnings

import numpy as np

import matplotlib
matplotlib.use('Agg')
from copy import deepcopy
from matplotlib import pyplot as plt

from pathlib import Path

from Simulation_Component.market import *
from Simulation_Component.agent import *
from helpers.helpers import *
from Simulation_Component.reporting import Reporting
from helpers.helper_classes import *

from torch.utils.data import DataLoader, TensorDataset

from datetime import datetime

### Def Logging:
import logging

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)


def simulation_normal(t, sim_time: int, simulation: Simulation, market: Market, supply_chain: Supply_Chain, sc_agent_list: list, cfg: json):
    logger.info("Start Simulation with the following settings: ")
    logger.info("Simulation Type: " + str(simulation.training_type))
    logger.info("Simulation Time: " + str(sim_time - t))

    t = t

    while t < sim_time:

        demand_t = np.array([market.split_demand_on_time(t)])
        # go over every level
        for i in range(len(sc_agent_list)):
            # go over every agent
            demand_per_agent = []
            shipment_per_agent = []
            for j, agent in enumerate(sc_agent_list[i]):
                demand, shipment = agent.act(demand_t[:, j], 0)
                demand_per_agent.append(demand)
                shipment_per_agent.append(shipment)
            demand_t = np.array(demand_per_agent)
            shipment_t = np.array(shipment_per_agent)
            shipment_sum_per_level = np.sum(shipment_t, axis=0)

            if i == 0:
                pass
            else:
                for k, agent in enumerate(sc_agent_list[i - 1]):
                    agent.set_shipment(shipment_sum_per_level[k])

        # set shipment for last level -> demand is always satisfied
            if i == (len(sc_agent_list) - 1):
                for k, agent in enumerate(sc_agent_list[-1]):
                    agent.set_shipment(demand_t[k][0])

        t += 1

def simulation_split_multichannel_lstm(t, sim_time: int, simulation: Simulation, market: Market, supply_chain: Supply_Chain, sc_agent_list: list, cfg: json):
    logger.info("Start Simulation with the following settings: ")
    logger.info("Simulation Type: " + str(simulation.training_type))
    logger.info("Simulation Time: " + str(sim_time))
    level = 1
    device = torch.device("cuda")
    t = t

    while t < sim_time:

        demand_t = np.array([market.split_demand_on_time(t)])
        # go over every level
        for i in range(len(sc_agent_list)):
            # go over every agent
            demand_per_agent = []
            shipment_per_agent = []
            if i != level:
                # go over every agent
                demand_per_agent = []
                shipment_per_agent = []
                for j, agent in enumerate(sc_agent_list[i]):
                    demand, shipment = agent.act(demand_t[:, j], 0)
                    demand_per_agent.append(demand)
                    shipment_per_agent.append(shipment)
                demand_t = np.array(demand_per_agent)
                shipment_t = np.array(shipment_per_agent)
                shipment_sum_per_level = np.sum(shipment_t, axis=0)

                if i == 0:
                    pass
                else:
                    for k, agent in enumerate(sc_agent_list[i - 1]):
                        agent.set_shipment(shipment_sum_per_level[k])

            # set shipment for last level -> demand is always satisfied
                if i == (len(sc_agent_list) - 1):
                    for k, agent in enumerate(sc_agent_list[-1]):
                        agent.set_shipment(demand_t[k][0])
                
            
            #######
            else:
                # implement inference during testing
                # 1) build features

                # go over each agent, train the local models and add it to the agent for testing phase
                logger.debug(f"Start simulation split mutltchannel lstm")

                datasets = []
                for k, agent in enumerate(sc_agent_list[1]): # change 1 to variable for dynamics
                    logger.debug(f"start building data for agent: {k}")

                    # build dataset and dataloader for training of agents defined
                    raw_demand_data = agent.demand_by_retailer_history
                    
                    data= []
                    for l in range(agent.num_retailer):
                        data.append(np.array(raw_demand_data[l]))
                    
                    df = pd.DataFrame(data).transpose()
                    df = df.iloc[-(agent.sequence_length-1):,:]
                    # df = pd.concat([df, demand_t[:, k]], ignore_index=True)
                    df.loc[len(df)] = demand_t[:, k] # ToDo Index is probably wrong
                    data_scaled = agent.forecasting_model.scaler.transform(df.to_numpy())

                    for l in range(agent.num_retailer):
                        # get data for retailer
                        train_data = data_scaled[-agent.sequence_length:,l].reshape(agent.sequence_length, 1)
                        datasets.append(train_data)
                # 2) inference
                
                # 2.1) get output lstm
                output_lstm_list = []
                num_ret_prev = 0
                for k, agent in enumerate(sc_agent_list[1]): # change 1 to variable for dynamics
                    

                    data = np.array(datasets[(k+num_ret_prev):(k+num_ret_prev+agent.num_retailer)])
                    num_ret_prev += 1

                    agent_lstm = agent.get_forecasting_model().lstm_model

                    for l, lstm in enumerate(agent_lstm):
                        lstm.eval()
                        input_tensor = torch.tensor([np.array(data[l])], dtype=torch.float32).to(device)
                        output_lstm_list.append(lstm(input_tensor))

                # 2.2) Feature Fusion
                fusion = torch.cat((output_lstm_list), axis = 2)

                # 2.3) get dense output
                output_dense_list = []
                for k, agent in enumerate(sc_agent_list[1]): # change 1 to variable for dynamics
                    agent_forecastingmodel = agent.get_forecasting_model()
                    agent_dense = agent_forecastingmodel.dense_model
                    for l, dense in enumerate(agent_dense):
                        dense.eval()
                        all_output = dense(fusion)
                        output = all_output[:, -1, :]
                        output_rescaled = (output.item()*agent_forecastingmodel.scaler.scale_[l])+agent_forecastingmodel.scaler.mean_[l]
                        
                        output_dense_list.append(output_rescaled)
                # print(output_dense_list)
                # 3) split prediction accros
                # go over every agent
                demand_per_agent = []
                shipment_per_agent = []
                for k, agent in enumerate(sc_agent_list[1]):
                    demand, shipment = agent.act_multichannel(demand_t[:, k], 0, output_dense_list[(k*agent.num_retailer):((k+1)*agent.num_retailer)])
                    demand_per_agent.append(demand)
                    shipment_per_agent.append(shipment)
                demand_t = np.array(demand_per_agent)
                shipment_t = np.array(shipment_per_agent)
                shipment_sum_per_level = np.sum(shipment_t, axis=0)

                if i == 0:
                    pass
                else:
                    for k, agent in enumerate(sc_agent_list[i - 1]):
                        agent.set_shipment(shipment_sum_per_level[k])

            # set shipment for last level -> demand is always satisfied
                if i == (len(sc_agent_list) - 1):
                    for k, agent in enumerate(sc_agent_list[-1]):
                        agent.set_shipment(demand_t[k][0])

        t += 1

def split_training_multichannel_lstm(simulation, market, supply_chain, sc_agent_list, cfg):
    logger.info("Starting Syncron Training")
    loss_cal = "aggregated" # individual or aggregated
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
    patience=400
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # 1) Create Models
    for i, agent in enumerate(sc_agent_list[1]):
        agent_key = "agent_"+str(i)
        agent_models[agent_key] = {} 
        lstm_models = []
        dense_models = []
        lstm_optimizers = []
        dense_optimizers = []
        dataloaders_id = [] 
        for j in range(agent.num_retailer):
            dataloader_id = i*agent.num_retailer+j
            dataloaders_id.append(dataloader_id)
            lstm = LSTM_Model(n_input =lstm_input_dim, n_output = lstm_output_dim, n_hidden=lstm_hidden_dim)
            lstm.to(device)
            lstm_optim = torch.optim.SGD(lstm.parameters(), lr=learning_rate, momentum=momentum)
            lstm_models.append(lstm)
            lstm_optimizers.append(lstm_optim)
            dense_input_dim = (agent.num_retailer*num_supplier)*lstm_hidden_dim
            dense = NetLocal2(n_input = dense_input_dim, n_output=dense_ouput_dim)
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

    for i, agent in enumerate(sc_agent_list[1]): # change 1 to variable for dynamics
        logger.info(f"start traing multichannel lstm for agent: {i}")

        # build dataset and dataloader for training of agents defined
        raw_demand_data = agent.demand_by_retailer_history
        
        data= []
        for i in range(agent.num_retailer):
            data.append(np.array(raw_demand_data[i]))
        
        df = pd.DataFrame(data).transpose()
        df_train = df.iloc[-(train_size+vall_size):-(vall_size), :]
        df_val = df.iloc[-(vall_size):, :]
        scaler =  StandardScaler()
        scaler = scaler.fit(df_train.to_numpy())
        scaler_list.append(scaler)
        data_train_scaled = scaler.transform(df_train)
        data_val_scaled = scaler.transform(df_val)
        
        agent_val_data = []
        for j in range(agent.num_retailer):
            # get data for retailer
            train_data = data_train_scaled[:,j].reshape(train_size, 1)
            val_data = data_val_scaled[:,j].reshape(vall_size, 1)
            X_train, y_train  = create_dataset(train_data, lookback=agent.sequence_length)
            X_val, y_val  = create_dataset(val_data, lookback=agent.sequence_length)
            dataloader = DataLoader(TensorDataset(X_train, y_train), 
                                    batch_size=agent.batch_size, shuffle=False, num_workers=1)
            trainloaders.append(dataloader)
            agent_val_data.append([X_val, y_val])
        val_data_list.append(agent_val_data)
        
    datasets = [None]*len(trainloaders)
    # 3) Training
    val_loss = []
    # train over every epoch
    for epoch in range(epochs):
        logger.info(f"Number Epoch: {epoch}")
        # go over every batch
        for batches in zip(*trainloaders):
            logger.debug("New Batch")

            if loss_cal == "aggregated":
                training_loss_epoch = [0]*(num_supplier)

            # set models to training mode
            for i, agent in enumerate(sc_agent_list[1]):
                agent_key = "agent_"+str(i)
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
            fusion = torch.cat(lstm_outputs_detached, axis = 2)

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
                    fusion_outputs = torch.cat(outputs, axis = 2)
                    fusion_targets = torch.cat(targets, axis = 2)
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

            #backward for lstm
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
            agent_key = "agent_"+str(i)
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
                    label_list.append(val_data_list[j][i][1][:, -1, :]) # use only last time step as target for validation loss

                    rescaled_labels = val_data_list[j][i][1][:, -1, :]*scaler_list[j].scale_[i]+scaler_list[j].scale_[i]
                    rescaled_label_list.append(rescaled_labels)
                    
                    output_detached = output.clone().detach().requires_grad_(True)
                    lstm_outputs.append(output)
                    lstm_outputs_detached.append(output_detached)

            # server routine
            logger.debug("Feature Fusion")
            fusion = torch.cat(lstm_outputs_detached, axis = 2)

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
                        outputs.append(dense_outputs[id][:, -1, :].cpu()*scaler_list[j].scale_[i]+scaler_list[j].scale_[i]) # use only last time step of predicted values for validation loss
                        targets.append(rescaled_label_list[id])
                    fusion_outputs = torch.cat(outputs, axis = 1)
                    fusion_targets = torch.cat(targets, axis = 1)
                    loss = loss_fn(fusion_outputs, fusion_targets)
                    loss_list.append(loss)
                    loss_list_item.append(loss.item())

            sum_val_loss = np.sum(loss_list_item)
            val_loss.append(sum_val_loss)

            logger.info(f"validation_loss_sum: {sum_val_loss}")
            logger.info(f"validation_loss_per_agent: {loss_list_item}")

            #ceck for early stopping
            early_stopping(sum_val_loss, agent_models)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
    best_models = early_stopping.best_model

    for j, agent_key in enumerate(best_models):
        lstm_models = best_models[agent_key]['lstm_model']
        dense_models = best_models[agent_key]['dense_model']

        model = MultiChannel_LSTM(num_channels = sc_agent_list[1][j].num_retailer, lstm_model=lstm_models, 
                                   dense_model=dense_models, scaler=scaler_list[j], device=device)

        sc_agent_list[1][j].set_forecasting_model(model)
        
    return val_loss

def local_training_multichannel_lstm(simulation, market, supply_chain, sc_agent_list, cfg):

    loss_cal = "aggregated" # individual or aggregated
    loss_fn = nn.L1Loss()
    device = select_gpu()
    val_loss_list = []

    for i, agent in enumerate(sc_agent_list[1]): # change 1 to variable for dynamics
        logger.info(f"Training agent: {i}")
        
        val_loss_agent = []

        lstm_input_dim = 1
        lstm_output_dim = 24
        lstm_hidden_dim = 48
        dense_input_dim = agent.num_retailer*lstm_hidden_dim
        dense_ouput_dim = 1

        learning_rate = 0.001
        momentum = 0.9

        train_size = simulation.train_size
        vall_size = simulation.val_size
        patience = 400
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        # 1) Create Model
        lstm_models = []
        dense_models = []
        lstm_optimizers = []
        dense_optimizers = []
        for j in range(agent.num_retailer):
            lstm = LSTM_Model(n_input =lstm_input_dim, n_output = lstm_output_dim, n_hidden=lstm_hidden_dim)
            lstm_optim = torch.optim.SGD(lstm.parameters(), lr=learning_rate, momentum=momentum)
            lstm_models.append(lstm)
            lstm_optimizers.append(lstm_optim)
            dense = NetLocal2(n_input = dense_input_dim, n_output=dense_ouput_dim)
            dense_optim = torch.optim.SGD(dense.parameters(), lr=learning_rate, momentum=momentum)
            dense_models.append(dense)
            dense_optimizers.append(dense_optim)
        # 2) Create Data
        trainloaders = []
        val_data_list = []

        # build dataset and dataloader for training of agents defined
        raw_demand_data = agent.demand_by_retailer_history
        print(len(raw_demand_data))
        
        data= []
        print("Num Retailer")
        print(agent.num_retailer)
        for j in range(agent.num_retailer):
            data.append(np.array(raw_demand_data[j]))
        
        df = pd.DataFrame(data).transpose()
        df_train = df.iloc[-(train_size+vall_size):-(vall_size), :]
        df_val = df.iloc[-(vall_size):, :]
        scaler =  StandardScaler()
        scaler = scaler.fit(df_train.to_numpy())
        data_train_scaled = scaler.transform(df_train)
        data_val_scaled = scaler.transform(df_val)

        # training data
        agent_val_data = []
        for j in range(agent.num_retailer):
            # get data for retailer
            train_data = data_train_scaled[:,j].reshape(train_size, 1)
            val_data = data_val_scaled[:,j].reshape(vall_size, 1)
            X_train, y_train  = create_dataset(train_data, lookback=agent.sequence_length)
            X_val, y_val  = create_dataset(val_data, lookback=agent.sequence_length)
            dataloader = DataLoader(TensorDataset(X_train, y_train), 
                                    batch_size=agent.batch_size, shuffle=False, num_workers=1)
            trainloaders.append(dataloader)
            val_data_list.append([X_val, y_val])
        
        datasets = [None]*len(trainloaders)


        # 3) Train

        for epoch in range(agent.epochs):
            logger.info(f"Number Epoch: {epoch}" )
            training_loss_epoch = [0]

            for model in lstm_models:
                model.to(device)
            for model in dense_models:
                model.to(device)

            # go over every batch
            for batches in zip(*trainloaders):
                logger.debug("New Batch")
                training_loss_epoch = [0]*(agent.num_retailer)

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
                    features_list.append(batch[0].to(device))
                    label_list.append(batch[1].to(device))

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
                fusion = torch.cat(lstm_outputs_detached, axis = 2)
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
                    fusion_outputs = torch.cat(dense_outputs, axis = 2)
                    fusion_targets = torch.cat(label_list, axis = 2)
                    loss_list = [loss_fn(fusion_outputs, fusion_targets)]*(agent.num_retailer)
                    loss_list_item = [loss_fn(fusion_outputs, fusion_targets).item()]*(agent.num_retailer)

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

                #backward for lstm
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
                    output = lstm_models[i](val_data_list[i][0].to(device))
                    output_detached = output.clone().detach().requires_grad_(True)
                    lstm_outputs.append(output)
                    lstm_outputs_detached.append(output_detached)

                    label_list.append(val_data_list[i][1][:, -1, :]) # use only last time step as target for validation loss

                    rescaled_labels = val_data_list[i][1][:, -1, :]*scaler.scale_[i]+scaler.mean_[i]
                    rescaled_label_list.append(rescaled_labels)


                # server routine
                logger.debug("Feature Fusion")
                fusion = torch.cat(lstm_outputs_detached, axis = 2)

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
                            loss = loss_fn(output[:, -1, :].cpu()*scaler.scale_[i]+scaler.mean_[i], rescaled_label_list[i])
                            loss_list.append(loss)
                            loss_list_item.append(loss.item())
                
                if loss_cal == "aggregated":
                    logger.debug("Loss aggregated")
                    outputs = []
                    targets = []
                    for i, output in enumerate(dense_outputs):
                        outputs.append(dense_outputs[i][:, -1, :].cpu()*scaler.scale_[i]+scaler.mean_[i]) # use only last time step of predicted values for validation loss
                        targets.append(rescaled_label_list[i])

                    fusion_outputs = torch.cat(outputs, axis = 1)
                    fusion_targets = torch.cat(targets, axis = 1)
                    loss = loss_fn(fusion_outputs, fusion_targets)
                    loss_list.append(loss)
                    loss_list_item.append(loss.item())

                sum_val_loss = np.sum(loss_list_item)
                # validation_loss.append(sum_val_loss)

                logger.info(f"validation_loss_sum: {sum_val_loss}")
                logger.info(f"validation_loss_per_agent: {loss_list_item}")

                #ceck for early stopping
                model_list = [lstm_models, dense_models]
                early_stopping(sum_val_loss, deepcopy(model_list))
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    epoch_diff = agent.epochs - epoch
                    #### if early stopping is reached, then use the last loss values for the rest of the planned epochs
                    for k in range(epoch_diff):
                        val_loss_agent.append(sum_val_loss)
                    break
                val_loss_agent.append(sum_val_loss)
        
        best_model = early_stopping.best_model
        lstm_models, dense_models = best_model
        logger.info(f"Training-Loss: {training_loss_epoch}")

        model = MultiChannel_LSTM(num_channels = agent.num_retailer, lstm_model=lstm_models, dense_model=dense_models, scaler=scaler, device=device)

        agent.set_forecasting_model(model)
        val_loss_list.append(val_loss_agent)
    
    # Convert to a NumPy array
    print(len(val_loss_list))
    print(len(val_loss_list[0]))
    print(len(val_loss_list[1]))
    print(len(val_loss_list[2]))
    val_loss = np.array(val_loss_list)

    # Sum over columns
    val_loss = np.sum(val_loss, axis=0)
    
    return val_loss


def run():
    np.random.seed(42)
    random.seed(42)
    script_directory = Path(__file__).parent

    # 1. load config
    config_path = script_directory / "config.yaml"

    logger.debug("The configuration is loaded from: ", config_path)

    # 2. initalize simulation
    simulation, market, supply_chain, sc_agent_list, cfg = init_simulaltion(config_path)

    # 3 run simulation
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    for run in range(simulation.sim_runs):

        logger.info(f"Iteration Number : {run + 1}/{simulation.sim_runs}")
        simulation, market, supply_chain, sc_agent_list = reset_simulaltion_from_dict(cfg)

        # 3.1 decide which simulation style: federated, local, None (use only MA)
        if simulation.training_type is None:
            sim_time = simulation.conv_time + simulation.sim_time + simulation.testing_time
            sim_start = 0
            simulation_normal(sim_start, sim_time, simulation, market, supply_chain, sc_agent_list, cfg)
            val_loss = None

        elif simulation.training_type == "local_multichannel":

            sim_time = simulation.conv_time + simulation.sim_time
            sim_start = 0
            simulation_normal(sim_start, sim_time, simulation, market, supply_chain, sc_agent_list, cfg)
        
            val_loss = local_training_multichannel_lstm(simulation, market, supply_chain, sc_agent_list, cfg)

            sim_start = sim_time
            sim_time = simulation.conv_time + simulation.sim_time + simulation.testing_time
            simulation_normal(sim_start, sim_time, simulation, market, supply_chain, sc_agent_list, cfg)
        
        elif simulation.training_type == "split_multichannel":

            sim_time = simulation.conv_time + simulation.sim_time
            sim_start = 0
            simulation_normal(sim_start, sim_time, simulation, market, supply_chain, sc_agent_list, cfg)

            val_loss = split_training_multichannel_lstm(simulation, market, supply_chain, sc_agent_list, cfg)

            sim_start = sim_time
            sim_time = simulation.conv_time + simulation.sim_time + simulation.testing_time
            simulation_split_multichannel_lstm(sim_start, sim_time, simulation, market, supply_chain, sc_agent_list, cfg)
            # simulation_normal(sim_start, sim_time, simulation, market, supply_chain, sc_agent_list, cfg)


        else:
            raise NotImplementedError(" Your Option is not implemented. To fix, please chose None, \"local\" or \"federated")

        # 4. create reporting
        reporting_path = script_directory / "Reporting"
        reporting_path.mkdir(parents=True, exist_ok=True)
        # Reporting(path="./Reporting").create_reporting(agent_list=sc_agent_list, market=market, supply_chain=supply_chain, cfg=cfg)
        Reporting(path= reporting_path, timestamp=timestamp).create_reporting_multiple_runs(agent_list=sc_agent_list,
                                                                                          market=market, supply_chain=supply_chain,
                                                                                          cfg=cfg, run_id=run, val_loss = val_loss)

if __name__ == "__main__":
    run()