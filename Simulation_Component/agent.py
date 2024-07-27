import json
import queue
import numpy as np

from scipy.special import softmax

from helpers.helper_classes import *


class Agent():
    """Agent within the Supply Chain
    """

    def __init__(
            self, id: int, sc_level: int, adjacency_matrix: np.array, lead_time_matrix: np.array, 
            training_time: int, cfg: json, cfg_all:json, 
            sequence_length: int, eopchs: int, batch_size: int, learning_rate: float, momentum: float) -> None:

        """Initialize the agent with all parameters

        Args:
            id (int): id of the agent within its supply chain level
            sc_level (int): supply chain level: 0 -> Retailer
            adjacency_matrix (np.array): array containing all connection between agents of this level with the next one
            lead_time_matrix (np.array): array containing either all leadtimes for individual agent-agent connections with
                                        this and the nexxt level or one lead time for all connections
            demand_hist_size (int): how many time steps should be considered for convergence phase and during training
            cfg (json): config file containing all additional configuration information of an agent
        """
        self.id = id  # id of agent within supply chain level
        self.sc_level = sc_level  # supply chain level the agent is on

        self.adjacency_matrix = adjacency_matrix  # adjaceny_matrix of this level with the next one
        self.lead_time_matrix = lead_time_matrix  # lead times of this level with the next one

        self.supplier_num = len(self.adjacency_matrix[self.id])  # get number of possible suppliers
        self.supplier_index = np.where(self.adjacency_matrix[self.id] == 1)[0]  # get ids of actula suppliers
        self.supplier_connections = len(self.supplier_index)  # get number of actual suppliers

        self.sequence_length = sequence_length
        self.epochs = eopchs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.cfg = cfg  # config file for the rest of the level
        self.cfg_all = cfg_all # config of everything
        if self.sc_level == 0:
            self.num_retailer = 1
        else:
            level_key = "sc_level_"+str(self.sc_level-1)
            adjacency_matrix_pre_level = self.cfg_all['sc_levels'][level_key]['adjaceny_list']
            self.num_retailer = len(np.where(np.array(adjacency_matrix_pre_level)[:, self.id]==1)[0])

        ### init replenishment
        self.replenish_strat = cfg['replenishment_strat'][self.id]  # inventory management
        self.__init_replenishment_strat()

        ### init forecasting
        self.forecasting_strat = cfg['forecasting_strat'][self.id]  # forecasting strat
        self.training_time = training_time  # how much history should be considered during convergence

        self.__init_forecasting_strat()  # set forecasting for convergence to Moveing Average

        ### init for reporting

        self.inventory_history = []
        self.order_history = []
        self.order_per_supplier_history = []
        self.received_shipment_history = []
        self.departed_shipment_history = []  # ToDo: Track
        self.demand_sum_history = []
        self.demand_by_retailer_history = []  
        for i in range(self.num_retailer):
            self.demand_by_retailer_history.append([])
        self.forecast_history = []
        self.forecast_by_retailer_history = []
        for i in range(self.num_retailer):
            self.forecast_by_retailer_history.append([])
        self.back_log_list = []  # not satisfied demands # ToDo: Track
        self.current_backlog = [0]*self.num_retailer

        self.variance_ratio = []

        ### init order mechanism: How the Order is split among the agents suppliers
        self.__init_splitting_meachanism()

        ### init flags for internal logic
        self.converging = True

        ## others
        self.current_demand = 0
        self.current_demand_sum = 0
        self.own_shipments = 0
        self.shipment_queue = queue.Queue(maxsize=lead_time_matrix + 1)
        self.__init_shipment_queue()

    def __init_shipment_queue(self) -> None:

        for i in range(self.lead_time_matrix):
            self.shipment_queue.put(0)

    def __init_replenishment_strat(self) -> None:
        """Sets the replenishment strategy using the inventory capacity, the initial inventory and
        strategy dependend parameters.
        Currently only the Order-Up-To (OUT) Strategy is implemented

        Raises:
            NotImplementedError: _description_
        """

        self.current_inv = self.cfg['init_inv'][self.id]
        self.inv_capacity = self.cfg['inv_capacity'][self.id]

        if (self.replenish_strat == "OUT"):
            self.R = self.cfg["R"][self.id]
            self.risk_factor = self.cfg['safety_risk_factor'][self.id]
            self.replenishment = OrderUpTo(R=self.R, lead_time=self.lead_time_matrix, risk_factor=self.risk_factor, inv_cap=self.inv_capacity)
        else:
            raise NotImplementedError

    def __init_forecasting_strat(self) -> None:
        """Inits the forecasting strategy for the convergence time at the start of the simulation
        """
        self.forecasting_model = MA(t=self.sequence_length)

    def __init_splitting_meachanism(self) -> None:
        """Initializes the true proportions of how the order is divided among the suppliers
        """

        gamma_list = np.random.normal(size=self.supplier_connections)
        self.gamma_list_normalized = softmax(gamma_list)
        
        # entered to keep same and equal splitting
        gamma_list = [(1/self.supplier_connections)]*self.supplier_connections
        self.gamma_list_normalized = gamma_list
        

    ### ToDo: make this dynamically and conceptualize logic
    def __build_prediction_feature(self) -> list:
        """Build the feature vector used for prediction

        Returns:
            list: contains the realization of the used features for prediction of the next demand
        """
        data = []
        for i in range(self.num_retailer):
            data.append(np.array(self.demand_by_retailer_history[i][-self.sequence_length:]))
        
        return data

    def _receive_shipment(self) -> None:
        """Receive the shipment and update inventory and reporting lists
        """

        self.inventory_history.append(self.current_inv)  # add current inv to history before shipment arrives
        shipment_size = self.shipment_queue.get() # shiopment arrives

        # check if old orders are available: only needed for the start fo the simulation
        self.current_inv = self.current_inv + shipment_size  # update current inventory

        # ToDo: Track possible wast
        if (self.current_inv > self.inv_capacity):  # if the order can not be stored - it is lost and the current inv is set to max capaity
            self.current_inv = self.inv_capacity
        self.received_shipment_history.append(shipment_size)  # add current shipment to shipment history
        
         # self.inventory_history.append(self.current_inv)  # add current inv after shipment arrives

    def _sell(self) -> None:
        """Sell the goods

        1) Track Demand
        2) Update inventory
        3) Track Backlog if neseccary

        """
        self.demand_sum_history.append(self.current_demand_sum)  # add current demand to demand history
        for i in range(self.num_retailer):
            self.demand_by_retailer_history[i].append(self.current_demand[i])
        new_inv = self.current_inv - (self.current_demand_sum + np.sum(self.current_backlog))  # update current inventory
        # if not all demand could be satisfied: save as backlog and add to demand of next period
        if new_inv < 0:
            self._split_own_shipments(new_inv)
        else:
            self.back_log_list.append(np.abs(self.current_backlog))
            self._create_own_shipments()
            self.current_inv = new_inv
            self.current_backlog = [0]*self.num_retailer
        
        
        # self.inventory_history.append(self.current_inv) # track current inv after selling own goods

        self.departed_shipment_history.append(self.own_shipments)

    def _create_own_shipments(self) -> None:
        """creates the shipments to satisfied demand
        """
        self.own_shipments = self.current_demand + self.current_backlog

    def _split_own_shipments(self, new_inv) -> None:
        """split shipment proportionally if it can not be satisfied fully
        """
        if self.current_demand_sum == 0:
            proportions = 0

        else:
            proportions = (self.current_demand + self.current_backlog) / (self.current_demand_sum + np.sum(self.current_backlog))
        
    
        self.own_shipments = np.trunc(self.current_inv * proportions)
        self.current_inv = self.current_inv - np.sum(self.own_shipments)
        self.current_backlog = (self.current_demand + self.current_backlog) - self.own_shipments
        self.back_log_list.append(np.abs(self.current_backlog))

    def _replenish_inv(self) -> None:
        """Replenish the Inventory by Forecasting Demand and Calculating the Order Size
        """
        data = self.__build_prediction_feature()  # get data for prediction


        d_est = self.forecasting_model.predict(data)  # forecast the demand

        for i in range(self.num_retailer):
            self.forecast_by_retailer_history[i].append(d_est[i])
        d_est_sum = np.sum(d_est)
        self.forecast_history.append(d_est_sum) # add current demand forecast to history of demand forecasts
        order = self.replenishment.compute_order_sum(d_est=d_est_sum, currnet_inv=self.current_inv)  # compute order size
        self.order_history.append(order)  # add order to order history
    
    def _replenish_inv_multichannel(self, predictions) -> None:
        """Replenish the Inventory by Forecasting Demand and Calculating the Order Size
        """
        data = self.__build_prediction_feature()  # get data for prediction


        d_est = predictions # forecast the demand

        for i in range(self.num_retailer):
            self.forecast_by_retailer_history[i].append(d_est[i])
        d_est_sum = np.sum(d_est)
        self.forecast_history.append(d_est_sum) # add current demand forecast to history of demand forecasts
        order = self.replenishment.compute_order_sum(d_est=d_est_sum, currnet_inv=self.current_inv)  # compute order size
        self.order_history.append(order)  # add order to order history

    def _split_orders(self) -> list:

        """Split the entire order among own suppliers.
        Use the true proportions and add some "error" to it. Then split the demand accordingly

        Returns:
            list: list containing the order size of each supplier of the next supply chain level
        """

        # chance spillting slighty ba adding a random error
        gamma_new = softmax(self.gamma_list_normalized + np.random.normal(loc=0, scale=0.01, size=self.supplier_connections))
        gamma_new = self.gamma_list_normalized
        demand = self.order_history[-1]
        demand_supplier = demand * np.array(gamma_new)

        # demand_list = np.zeros((1, self.supplier_num))[0]
        demand_list = [0] * self.supplier_num

        d = 0
        for i in range(len(demand_supplier) - 1):
            demand_supplier[i] = np.round(demand_supplier[i])
            d = np.round(demand_supplier[i]) + d

        demand_supplier[-1] = demand - d

        for i, ind_ in enumerate(self.supplier_index):
            demand_list[ind_] = demand_supplier[i]

        self.order_per_supplier_history.append(demand_list)

        return demand_list

    def act(self, demand_t: int, sum_received_shipments_t: int) -> list:
        """Based the current demand during time step t,
        run through the agents actions and return the order size per supplier.

        Args:
            demand_t (int): overall demand during time step t

        Returns:
            list: order size for each supplier of the next level
        """
        self.current_demand = demand_t
        self.current_demand_sum = np.sum(demand_t)
        self._receive_shipment()
        self._sell()
        self._replenish_inv()
        return self._split_orders(), self.own_shipments # order for own supplier, and shipments to own retailer
    
    def act_multichannel(self, demand_t: int, sum_received_shipments_t: int, predictions) -> list:
        """Based the current demand during time step t,
        run through the agents actions and return the order size per supplier.

        Args:
            demand_t (int): overall demand during time step t

        Returns:
            list: order size for each supplier of the next level
        """
        self.current_demand = demand_t
        self.current_demand_sum = np.sum(demand_t)
        self._receive_shipment()
        self._sell()
        self._replenish_inv_multichannel(predictions)
        return self._split_orders(), self.own_shipments

    # setter

    def set_converging(self, flag: bool) -> None:
        """Set Convering to singal if the simulation is done with its convergence phase

        Args:
            flag (bool): True: still converging; False: done with converging
        """

        self.converging = flag

    def set_forecasting_model(self, forecasting_model: Forecasting) -> None:
        """Set the forecasting model

        Args:
            forecasting_model (Forecasting): set the current forecasting model
        """
        self.forecasting_model = forecasting_model
    
    def get_forecasting_model(self) -> Forecasting:

        return self.forecasting_model

    def set_shipment(self, shipment: int) -> None:

        self.shipment_queue.put(shipment)

    def get_variance_ratio(self, last_t: int) -> float:

        var_demand = np.var(self.demand_sum_history[-last_t:])
        var_orders = np.var(self.order_history[-last_t:])

        return np.round((var_orders / var_demand), 2)

    def set_variance_ratio(self, last_t: int) -> None:

        self.variance_ratio.append(self.get_variance_ratio(last_t))

