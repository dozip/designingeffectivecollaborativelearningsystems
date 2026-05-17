from typing import Optional

import numpy as np
import pandas as pd
import torch

from .base import ForecastingBackend
from . import register_backend


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
        from main import split_training_multichannel_lstm
        return split_training_multichannel_lstm(
            simulation, market, supply_chain, sc_agent_list, self.cfg
        )

    @property
    def collaborative_level(self) -> Optional[int]:
        return 1

    def collaborative_predict(self, level_agents, demand_t, sc_agent_list, t) -> list:
        device = torch.device("cuda")

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
        num_ret_prev = 0
        for k, agent in enumerate(level_agents):
            data = np.array(datasets[(k + num_ret_prev):(k + num_ret_prev + agent.num_retailer)])
            num_ret_prev += 1

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
