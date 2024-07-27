import math
import random
import numpy as np
from scipy.special import softmax
from abc import ABC, abstractmethod

class Market(ABC):

    @abstractmethod
    def demand_funtion(self):
        pass

    @abstractmethod
    def get_demand(self, t):
        pass

    @abstractmethod
    def _calcrho(self, rho, rho1, rho2):
        pass

    @abstractmethod
    def _init_splitting_meachanism(self):
        pass

    @abstractmethod
    def split_demand_on_time(self, t):
        pass

    @abstractmethod
    def split_demand(self, demand_t):
        pass

    @abstractmethod
    def act(self, demand_t):
        pass

class MarketArtifical(Market):
    """Market class containing all parameters of the market, the demand and market share
    """

    def __init__(self, sim_time: int, primary_demand: int, trend_mag: float, seasonality_mag: float, seasonality_freq: int, random_walk_mean,
                 random_walk_var, retailer_num, market_demand_split) -> None:
        self.sim_time = sim_time # time of the entire simulation --> conv + sim + testing time
        self.primary_demand = primary_demand
        self.trend_mag = trend_mag
        self.seasonality_mag = seasonality_mag
        self.seasonality_freq = seasonality_freq
        self.random_walk_mean = random_walk_mean
        self.random_walk_var = random_walk_var

        self.retailer_num = retailer_num
        self.market_demand_split = np.array(market_demand_split)

        self.demand = None
        self.demand_funtion()

        # self._init_splitting_meachanism()

        self.split_demand_history = []

    def demand_funtion(self):
        time = np.arange(self.sim_time)
        b = 2 * math.pi / self.seasonality_freq
        self.demand = np.round(self.primary_demand + self.trend_mag * time + self.seasonality_mag * np.sin(b * time) +
                               np.random.normal(self.random_walk_mean, self.random_walk_var, size=self.sim_time), 0) ## see Zhao, X., Xie, J., Leung, J., 2002. The impact of forecasting model selection on the value of information sharing in a supply chain. European Journal of Operational Research 142, 321–344.
        
        self.demand = np.round((self.primary_demand + self.trend_mag * time) * ((self.seasonality_mag + np.sin(((2 * math.pi) / 52) * time))/(self.seasonality_mag)) + self.random_walk_mean * np.random.normal(loc=0, scale=1, size=self.sim_time), 0) # Bayraktar, E., Lenny Koh, S. C., Gunasekaran, A., Sari, K., & Tatoglu, E. (2008). The role of forecasting on bullwhip effect for E-SCM applications. International Journal of Production Economics, 113(1), 193–204. doi:10.1016/j.ijpe.2007.03.024

        self.demand = np.round(self.primary_demand + self.trend_mag * time + self.seasonality_mag * np.sin(((2 * math.pi)/self.seasonality_freq)*time) + self.random_walk_mean * np.random.normal(loc=0, scale=1, size=self.sim_time), 0) # see Zhao, X., Xie, J., Leung, J., 2002. The impact of forecasting model selection on the value of information sharing in a supply chain. European Journal of Operational Research 142, 321–344.
        
        # self.demand = np.round(self.primary_demand)

    def get_demand(self, t):
        return self.demand[t]

    def _calcrho(self, rho, rho1, rho2):
        part_1 = 1- (rho1*rho2)
        part_2 = np.sqrt((1-math.pow(rho1,2))*(1-math.pow(rho2,2)))

        calcrho_ = rho*(part_1/part_2)

        return calcrho_

    def _init_splitting_meachanism(self):
        # theta_list = np.random.normal(size=self.retailer_num)
        self.theta_list = np.array([0.75, 0.25])
        # self.theta_list_normalized = softmax(theta_list)

    def split_demand_on_time(self, t):
        # theta_new = softmax(self.theta_list_normalized +
        #                     np.random.normal(loc=0, scale=0.01, size=self.retailer_num))
        # theta_new = np.array([0.75, 0.25])
        demand = self.get_demand(t)
        demand_list = demand * self.market_demand_split

        d = 0
        for i in range(len(demand_list) - 1):
            demand_list[i] = np.round(demand_list[i])
            d = np.round(demand_list[i]) + d

        demand_list[-1] = demand - d

        self.split_demand_history.append(demand_list)
        res = []
        for d in demand_list:
            res.append(d)

        return res

    def split_demand(self, demand_t):
        # theta_new = softmax(self.theta_list_normalized +
        #                     np.random.normal(loc=0, scale=0.01, size=self.retailer_num))
        # theta_new = [0.75, 0.25]
        demand_list = demand_t * self.market_demand_split

        d = 0
        for i in range(len(demand_list) - 1):
            demand_list[i] = np.round(demand_list[i])
            d = np.round(demand_list[i]) + d

        demand_list[-1] = demand_t - d

        self.split_demand_history.append(demand_list)

        return [demand_list]

    def act(self, demand_t):
        demand = self.split_demand(demand_t=demand_t)
        res = []
        for d in demand:
            res.append(d)
        return res


class MarketDataSource(Market):

    def __init__(self, data, retailer_num, market_demand_split) -> None:
        self.retailer_num = retailer_num
        self.market_demand_split = np.array(market_demand_split)

        self.demand = None
        self.data = data
        self.demand_funtion()

        # self._init_splitting_meachanism()

        self.split_demand_history = []

    def demand_funtion(self):
        self.demand = self.data

    def get_demand(self, t):
        return self.demand[t]
    
    def _calcrho(self, rho, rho1, rho2):
        part_1 = 1- (rho1*rho2)
        part_2 = np.sqrt((1-math.pow(rho1,2))*(1-math.pow(rho2,2)))

        calcrho_ = rho*(part_1/part_2)

        return calcrho_
    
    def _init_splitting_meachanism(self):
        self.theta_list = np.array([0.75, 0.25])

    def split_demand_on_time(self, t):
        demand = self.get_demand(t)
        demand_list = demand * self.market_demand_split

        d = 0
        for i in range(len(demand_list) - 1):
            demand_list[i] = np.round(demand_list[i])
            d = np.round(demand_list[i]) + d

        demand_list[-1] = demand - d

        self.split_demand_history.append(demand_list)
        res = []
        for d in demand_list:
            res.append(d)

        return res

    def split_demand(self, demand_t):
        demand_list = demand_t * self.market_demand_split

        d = 0
        for i in range(len(demand_list) - 1):
            demand_list[i] = np.round(demand_list[i])
            d = np.round(demand_list[i]) + d

        demand_list[-1] = demand_t - d

        self.split_demand_history.append(demand_list)

        return [demand_list]

    def act(self, demand_t):
        demand = self.split_demand(demand_t=demand_t)
        res = []
        for d in demand:
            res.append(d)
        return res
