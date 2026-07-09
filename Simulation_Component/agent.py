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
            training_time: int, cfg: json, cfg_all: json,
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

        # Keep lead time as a plain int. This avoids queue maxsize/range issues
        # if a numpy scalar is passed in.
        self.lead_time_matrix = int(np.asarray(lead_time_matrix).item())

        self.supplier_num = len(self.adjacency_matrix[self.id])  # get number of possible suppliers
        self.supplier_index = np.where(self.adjacency_matrix[self.id] == 1)[0]  # get ids of actula suppliers
        self.supplier_connections = len(self.supplier_index)  # get number of actual suppliers

        self.sequence_length = sequence_length
        self.epochs = eopchs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.cfg = cfg  # config file for the rest of the level
        self.cfg_all = cfg_all  # config of everything
        if self.sc_level == 0:
            self.num_retailer = 1
        else:
            level_key = "sc_level_" + str(self.sc_level - 1)
            adjacency_matrix_pre_level = self.cfg_all['sc_levels'][level_key]['adjaceny_list']
            self.num_retailer = len(np.where(np.array(adjacency_matrix_pre_level)[:, self.id] == 1)[0])

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

        # FIX:
        # Keep demand/backlog/outgoing shipments consistently as vectors of
        # length self.num_retailer. Previously these could become scalars
        # in zero-demand or initialization cases, which later produced ragged
        # shipment_per_agent lists in runner.py.
        self.current_backlog = np.zeros(self.num_retailer, dtype=float)
        self.current_demand = np.zeros(self.num_retailer, dtype=float)
        self.current_demand_sum = 0.0
        self.own_shipments = np.zeros(self.num_retailer, dtype=float)

        self.variance_ratio = []

        ### init order mechanism: How the Order is split among the agents suppliers
        self.__init_splitting_meachanism()

        ### init flags for internal logic
        self.converging = True

        ## others
        # Incoming shipments replenish this agent's own inventory, which is scalar.
        # The outgoing own_shipments remain vector-valued.
        self.shipment_queue = queue.Queue(maxsize=self.lead_time_matrix + 1)
        self.__init_shipment_queue()

    # ---------------------------------------------------------------------
    # Shape normalization helpers
    # ---------------------------------------------------------------------

    def _zero_vector(self) -> np.ndarray:
        """Return a zero vector with one entry per downstream retailer/customer."""
        return np.zeros(self.num_retailer, dtype=float)

    def _as_vector(self, value, name: str = "value") -> np.ndarray:
        """Normalize demand/backlog/outgoing-shipment values to shape (num_retailer,).

        Scalars are accepted when num_retailer == 1. For agents with multiple
        downstream retailers, only scalar zero is accepted and expanded to a
        zero vector. A non-zero scalar would be ambiguous and is therefore
        rejected with a clear error.
        """
        arr = np.asarray(value, dtype=float)

        if arr.ndim == 0:
            scalar = float(arr)
            if self.num_retailer == 1:
                return np.array([scalar], dtype=float)
            if scalar == 0.0:
                return self._zero_vector()
            raise ValueError(
                f"Agent {self.id} level {self.sc_level}: {name} is a non-zero scalar "
                f"({scalar}) but num_retailer={self.num_retailer}. "
                "Cannot infer how to distribute it."
            )

        arr = arr.reshape(-1)

        if arr.size == self.num_retailer:
            return arr.astype(float)

        if arr.size == 1:
            scalar = float(arr[0])
            if self.num_retailer == 1:
                return np.array([scalar], dtype=float)
            if scalar == 0.0:
                return self._zero_vector()

        raise ValueError(
            f"Agent {self.id} level {self.sc_level}: {name} has shape "
            f"{np.shape(value)}, expected ({self.num_retailer},). value={value}"
        )

    def _as_scalar(self, value, name: str = "value") -> float:
        """Normalize incoming inventory shipments to a scalar.

        Incoming shipments update current_inv, which is scalar. If a vector is
        provided, the entries are summed.
        """
        arr = np.asarray(value, dtype=float)

        if arr.ndim == 0:
            return float(arr)

        return float(np.sum(arr))

    def __init_shipment_queue(self) -> None:
        # Incoming shipments are scalar inventory arrivals. Use float zeros
        # consistently instead of int zeros.
        for i in range(self.lead_time_matrix):
            self.shipment_queue.put(0.0)

    def __init_replenishment_strat(self) -> None:
        """Sets the replenishment strategy using the inventory capacity, the initial inventory and
        strategy dependend parameters.
        Currently only the Order-Up-To (OUT) Strategy is implemented

        Raises:
            NotImplementedError: _description_
        """

        self.current_inv = float(self.cfg['init_inv'][self.id])
        self.inv_capacity = float(self.cfg['inv_capacity'][self.id])

        if self.replenish_strat == "OUT":
            self.R = self.cfg["R"][self.id]
            self.risk_factor = self.cfg['safety_risk_factor'][self.id]
            self.replenishment = OrderUpTo(
                R=self.R,
                lead_time=self.lead_time_matrix,
                risk_factor=self.risk_factor,
                inv_cap=self.inv_capacity
            )
        else:
            raise NotImplementedError

    def __init_forecasting_strat(self) -> None:
        """Inits the forecasting strategy for the convergence time at the start of the simulation
        """
        self.forecasting_model = MA(t=self.sequence_length)

    def __init_splitting_meachanism(self) -> None:
        """Initialize how this agent divides orders among its suppliers.

        Default behavior is unchanged: orders are split equally among all
        connected suppliers. If supply_chain.dynamic_supplier_allocation.enabled
        is true in the config, the equal split becomes the long-run baseline,
        but small bounded and smoothed deviations are allowed over time.
        """

        if self.supplier_connections <= 0:
            self.gamma_base = np.array([], dtype=float)
            self.gamma_current = np.array([], dtype=float)
            self.gamma_list_normalized = np.array([], dtype=float)
        else:
            self.gamma_base = np.array(
                [(1 / self.supplier_connections)] * self.supplier_connections,
                dtype=float,
            )
            self.gamma_current = self.gamma_base.copy()

            # Backward-compatible name used by the old splitting logic.
            self.gamma_list_normalized = self.gamma_base.copy()

        # Optional diagnostics: stores the supplier-allocation vector used in
        # each call to _split_orders(). This is useful for checking whether the
        # dynamic option actually changes the upstream demand correlations.
        self.supplier_allocation_history = []

    def _get_dynamic_supplier_allocation_cfg(self) -> dict:
        """Read dynamic supplier-allocation options from the config.

        Expected YAML location when cfg_all is the supply_chain config:

            supply_chain:
              dynamic_supplier_allocation:
                enabled: false

        The function also supports the same block one level higher, in case
        cfg_all is later changed to the full merged config.
        """

        if not isinstance(self.cfg_all, dict):
            return {}

        cfg = self.cfg_all.get("dynamic_supplier_allocation", None)

        if cfg is None and isinstance(self.cfg_all.get("supply_chain"), dict):
            cfg = self.cfg_all["supply_chain"].get("dynamic_supplier_allocation", None)

        if cfg is None:
            return {}

        if isinstance(cfg, bool):
            return {"enabled": cfg}

        if isinstance(cfg, dict):
            return cfg

        return {}

    def _as_bool(self, value, default: bool = False) -> bool:
        """Parse booleans from bools/strings/numbers in YAML configs."""

        if value is None:
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}

        return bool(value)

    def _dynamic_supplier_allocation_is_enabled(self, cfg: dict) -> bool:
        """Return whether dynamic supplier allocation should be used here."""

        enabled = self._as_bool(cfg.get("enabled", False), default=False)

        if not enabled:
            return False

        # With one or zero suppliers, dynamic allocation cannot change anything.
        if self.supplier_connections <= 1:
            return False

        # Optional restriction, for example apply_to_sc_levels: [0].
        levels = cfg.get("apply_to_sc_levels", cfg.get("levels", None))

        if levels is not None:
            if not isinstance(levels, (list, tuple, set)):
                levels = [levels]

            try:
                levels = {int(level) for level in levels}
            except (TypeError, ValueError):
                levels = set()

            if self.sc_level not in levels:
                return False

        return True

    def _float_cfg(self, cfg: dict, key: str, default: float) -> float:
        """Read a float config value with a safe default."""

        try:
            return float(cfg.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _bounded_simplex_normalize(
            self,
            values,
            min_share: float,
            max_share: float,
    ) -> np.ndarray:
        """Normalize supplier shares to sum to one while respecting bounds.

        The bounds are treated as safeguards against unrealistic jumps. If a
        bound combination is infeasible for the number of suppliers, it is
        relaxed automatically.
        """

        n = self.supplier_connections

        if n <= 0:
            return np.array([], dtype=float)

        base = self.gamma_base.copy()
        values = np.asarray(values, dtype=float).reshape(-1)

        if values.size != n or not np.all(np.isfinite(values)):
            return base

        values = np.maximum(values, 0.0)

        if float(np.sum(values)) <= 0.0:
            return base

        values = values / float(np.sum(values))

        # Make bounds feasible.
        min_share = max(0.0, float(min_share))
        max_share = min(1.0, float(max_share))

        if min_share * n > 1.0:
            min_share = 0.0

        if max_share * n < 1.0:
            max_share = 1.0

        if min_share > max_share:
            min_share = 0.0
            max_share = 1.0

        x = np.clip(values, min_share, max_share)

        # Iteratively distribute the residual mass over entries not stuck at a
        # bound. For the small supplier counts here this is enough and stable.
        for _ in range(20):
            diff = 1.0 - float(np.sum(x))

            if abs(diff) < 1e-12:
                break

            if diff > 0:
                free = x < (max_share - 1e-12)
            else:
                free = x > (min_share + 1e-12)

            if not np.any(free):
                break

            x[free] += diff / int(np.sum(free))
            x = np.clip(x, min_share, max_share)

        # Final fallback if numerical edge cases remain.
        if float(np.sum(x)) <= 0.0 or not np.all(np.isfinite(x)):
            return base

        x = x / float(np.sum(x))

        return x

    def _get_supplier_allocation(self) -> np.ndarray:
        """Return supplier shares for the current order-splitting step.

        Default: fixed equal allocation.
        Optional: bounded and temporally smoothed allocation around equal shares.
        """

        cfg = self._get_dynamic_supplier_allocation_cfg()

        if not self._dynamic_supplier_allocation_is_enabled(cfg):
            return self.gamma_base.copy()

        n = self.supplier_connections
        default_min_share = 0.5 / n
        default_max_share = min(1.0, 1.5 / n)

        noise_std = self._float_cfg(cfg, "noise_std", 0.02)
        smoothing = self._float_cfg(cfg, "smoothing", 0.90)
        min_share = self._float_cfg(cfg, "min_share", default_min_share)
        max_share = self._float_cfg(cfg, "max_share", default_max_share)

        # Keep smoothing in a safe range. Higher means slower changes.
        smoothing = min(max(smoothing, 0.0), 0.999)
        noise_std = max(noise_std, 0.0)

        # Perturb log-shares rather than shares directly. This keeps the
        # allocation positive before applying explicit safety bounds.
        candidate = softmax(
            np.log(self.gamma_base + 1e-12)
            + np.random.normal(loc=0.0, scale=noise_std, size=n)
        )

        candidate = self._bounded_simplex_normalize(
            candidate,
            min_share=min_share,
            max_share=max_share,
        )

        gamma_new = (
            smoothing * self.gamma_current
            + (1.0 - smoothing) * candidate
        )

        gamma_new = self._bounded_simplex_normalize(
            gamma_new,
            min_share=min_share,
            max_share=max_share,
        )

        self.gamma_current = gamma_new.copy()

        return gamma_new

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

        self.inventory_history.append(float(self.current_inv))  # add current inv to history before shipment arrives

        shipment_size = self.shipment_queue.get()  # shipment arrives
        shipment_size = self._as_scalar(shipment_size, name="received shipment")

        # check if old orders are available: only needed for the start of the simulation
        self.current_inv = float(self.current_inv) + shipment_size  # update current inventory

        # ToDo: Track possible waste
        if self.current_inv > self.inv_capacity:  # if the order can not be stored - it is lost and current inv is max capacity
            self.current_inv = self.inv_capacity

        self.received_shipment_history.append(shipment_size)  # add current shipment to shipment history

        # self.inventory_history.append(self.current_inv)  # add current inv after shipment arrives

    def _sell(self) -> None:
        """Sell the goods

        1) Track Demand
        2) Update inventory
        3) Track Backlog if necessary
        """
        self.current_demand = self._as_vector(self.current_demand, name="current_demand")
        self.current_backlog = self._as_vector(self.current_backlog, name="current_backlog")
        self.current_demand_sum = float(np.sum(self.current_demand))

        self.demand_sum_history.append(self.current_demand_sum)  # add current demand to demand history

        for i in range(self.num_retailer):
            self.demand_by_retailer_history[i].append(float(self.current_demand[i]))

        total_required = self.current_demand_sum + float(np.sum(self.current_backlog))
        new_inv = float(self.current_inv) - total_required  # update current inventory

        # if not all demand could be satisfied: save as backlog and add to demand of next period
        if new_inv < 0:
            self._split_own_shipments(new_inv)
        else:
            self.back_log_list.append(np.abs(self.current_backlog).copy())
            self._create_own_shipments()
            self.current_inv = float(new_inv)
            self.current_backlog = self._zero_vector()

        # self.inventory_history.append(self.current_inv) # track current inv after selling own goods

        self.own_shipments = self._as_vector(self.own_shipments, name="own_shipments")
        self.departed_shipment_history.append(self.own_shipments.copy())

    def _create_own_shipments(self) -> None:
        """creates the shipments to satisfied demand
        """
        self.current_demand = self._as_vector(self.current_demand, name="current_demand")
        self.current_backlog = self._as_vector(self.current_backlog, name="current_backlog")
        self.own_shipments = self.current_demand + self.current_backlog

    def _split_own_shipments(self, new_inv) -> None:
        """split shipment proportionally if it can not be satisfied fully
        """
        self.current_demand = self._as_vector(self.current_demand, name="current_demand")
        self.current_backlog = self._as_vector(self.current_backlog, name="current_backlog")

        total_required = self.current_demand_sum + float(np.sum(self.current_backlog))

        # FIX:
        # Previously this branch used proportions = 0, which made own_shipments
        # a scalar. Always keep it vector-shaped.
        if total_required <= 0:
            proportions = self._zero_vector()
        else:
            proportions = (self.current_demand + self.current_backlog) / total_required

        available_inventory = max(float(self.current_inv), 0.0)
        self.own_shipments = np.trunc(available_inventory * proportions)
        self.own_shipments = self._as_vector(self.own_shipments, name="own_shipments")

        self.current_inv = float(self.current_inv) - float(np.sum(self.own_shipments))
        self.current_backlog = (self.current_demand + self.current_backlog) - self.own_shipments
        self.current_backlog = self._as_vector(self.current_backlog, name="current_backlog")

        self.back_log_list.append(np.abs(self.current_backlog).copy())

    def _replenish_inv(self) -> None:
        """Replenish the Inventory by Forecasting Demand and Calculating the Order Size
        """
        data = self.__build_prediction_feature()  # get data for prediction

        d_est = self.forecasting_model.predict(data)  # forecast the demand
        d_est = self._as_vector(d_est, name="forecast d_est")

        for i in range(self.num_retailer):
            self.forecast_by_retailer_history[i].append(float(d_est[i]))

        d_est_sum = float(np.sum(d_est))
        self.forecast_history.append(d_est_sum)  # add current demand forecast to history of demand forecasts

        order = self.replenishment.compute_order_sum(
            d_est=d_est_sum,
            currnet_inv=self.current_inv
        )  # compute order size
        self.order_history.append(order)  # add order to order history

    def _replenish_inv_multichannel(self, predictions) -> None:
        """Replenish the Inventory by Forecasting Demand and Calculating the Order Size
        """
        data = self.__build_prediction_feature()  # get data for prediction

        d_est = predictions  # forecast the demand
        d_est = self._as_vector(d_est, name="multichannel predictions")

        for i in range(self.num_retailer):
            self.forecast_by_retailer_history[i].append(float(d_est[i]))

        d_est_sum = float(np.sum(d_est))
        self.forecast_history.append(d_est_sum)  # add current demand forecast to history of demand forecasts

        order = self.replenishment.compute_order_sum(
            d_est=d_est_sum,
            currnet_inv=self.current_inv
        )  # compute order size
        self.order_history.append(order)  # add order to order history

    def _split_orders(self) -> list:
        """Split the current order among this agent's suppliers.

        By default, this reproduces the old behavior exactly: fixed equal
        allocation across connected suppliers. If dynamic supplier allocation is
        enabled in the config, the allocation is allowed to deviate slightly and
        smoothly from equal allocation.

        Returns:
            list: list containing the order size of each supplier of the next supply chain level
        """

        gamma_new = self._get_supplier_allocation()
        self.supplier_allocation_history.append(gamma_new.copy())

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
        self.current_demand = self._as_vector(demand_t, name="demand_t")
        self.current_demand_sum = float(np.sum(self.current_demand))

        self._receive_shipment()
        self._sell()
        self._replenish_inv()

        self.own_shipments = self._as_vector(self.own_shipments, name="own_shipments")
        return self._split_orders(), self.own_shipments.copy()  # order for own supplier, and shipments to own retailer

    def act_multichannel(self, demand_t: int, sum_received_shipments_t: int, predictions) -> list:
        """Based the current demand during time step t,
        run through the agents actions and return the order size per supplier.

        Args:
            demand_t (int): overall demand during time step t

        Returns:
            list: order size for each supplier of the next level
        """
        self.current_demand = self._as_vector(demand_t, name="demand_t")
        self.current_demand_sum = float(np.sum(self.current_demand))

        self._receive_shipment()
        self._sell()
        self._replenish_inv_multichannel(predictions)

        self.own_shipments = self._as_vector(self.own_shipments, name="own_shipments")
        return self._split_orders(), self.own_shipments.copy()

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
        # Incoming shipment replenishes scalar inventory. If a vector arrives,
        # sum it before putting it into the queue.
        self.shipment_queue.put(self._as_scalar(shipment, name="set_shipment shipment"))

    def get_variance_ratio(self, last_t: int) -> float:

        var_demand = np.var(self.demand_sum_history[-last_t:])
        var_orders = np.var(self.order_history[-last_t:])

        return np.round((var_orders / var_demand), 2)

    def set_variance_ratio(self, last_t: int) -> None:

        self.variance_ratio.append(self.get_variance_ratio(last_t))
