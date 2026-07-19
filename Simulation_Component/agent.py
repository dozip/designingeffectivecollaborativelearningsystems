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
            sequence_length: int, eopchs: int, batch_size: int, learning_rate: float, momentum: float,
            num_products: int = 1, product_to_manufacturer=None) -> None:

        """Initialize the agent with all parameters

        Args:
            id (int): id of the agent within its supply chain level
            sc_level (int): supply chain level: 0 -> Retailer
            adjacency_matrix (np.array): array containing all connection between agents of this level with the next one
            lead_time_matrix (np.array): array containing either all leadtimes for individual agent-agent connections with
                                        this and the nexxt level or one lead time for all connections
            training_time (int): number of historical time steps made available to the initial forecasting strategy during the convergence phase
            cfg (json): config file containing all additional configuration information of an agent
            num_products (int): number of distinct products in the market. Retailers hold
                separate inventory / forecast / order per product; upstream agents each
                produce exactly one product (num_products collapses to 1 for them).
            product_to_manufacturer: optional routing map product_index -> supplier(s).
                Only used by retailers (sc_level 0). None => equal split across all
                connected suppliers per product, which reproduces the legacy gamma split.
        """
        self.id = id  # id of agent within supply chain level
        self.sc_level = sc_level  # supply chain level the agent is on

        self.adjacency_matrix = adjacency_matrix  # adjaceny_matrix of this level with the next one

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

        # -----------------------------------------------------------------
        # Product / channel axes.
        #
        # Two distinct axes exist on an agent:
        #   num_products : the inventory / order / replenishment axis. Retailers
        #                  carry one per product; single-product upstream agents
        #                  collapse this to 1.
        #   num_retailer : the forecasting-channel axis (one incoming demand
        #                  stream per channel). For a retailer the channels ARE
        #                  the products, so num_retailer == num_products. For a
        #                  manufacturer the channels are the retailers feeding it.
        #
        # channel_to_product maps each forecasting channel to the product whose
        # inventory it feeds. Retailer: identity (channel p -> product p).
        # Manufacturer: every channel feeds the single product 0.
        # -----------------------------------------------------------------
        if self.sc_level == 0:
            self.num_products = int(num_products)
            self.num_retailer = self.num_products
            self.channel_to_product = np.arange(self.num_products, dtype=int)
        else:
            self.num_products = 1
            level_key = "sc_level_" + str(self.sc_level - 1)
            adjacency_matrix_pre_level = self.cfg_all['sc_levels'][level_key]['adjaceny_list']
            self.num_retailer = len(np.where(np.array(adjacency_matrix_pre_level)[:, self.id] == 1)[0])
            self.channel_to_product = np.zeros(self.num_retailer, dtype=int)

        # Precompute, per product, the list of channels feeding it.
        self._channels_of_product = [
            np.where(self.channel_to_product == p)[0]
            for p in range(self.num_products)
        ]

        # Per-product lead times. Each product may ship from a different
        # supplier (manufacturer) with its own lead time. Falls back to a single
        # broadcast lead time, which reproduces the single-product behaviour.
        self.lead_time_per_product = self._build_lead_times(lead_time_matrix)
        # Representative scalar lead time kept for backward compatibility.
        self.lead_time_matrix = int(self.lead_time_per_product[0])

        ### init replenishment
        self.replenish_strat = cfg['replenishment_strat'][self.id]  # inventory management
        self.__init_replenishment_strat()

        ### init forecasting
        self.forecasting_strat = cfg['forecasting_strat'][self.id]  # forecasting strat
        self.training_time = training_time  # how much history should be considered during convergence

        self.__init_forecasting_strat()  # set forecasting for convergence to Moveing Average

        ### init for reporting
        # Flat reporting lists. For multi-product agents these hold the
        # aggregate (sum over products) per time step, so single-product runs
        # are byte-identical. Per-product detail lives in the *_by_product
        # lists below.
        self.inventory_history = []
        self.order_history = []
        self.order_per_supplier_history = []
        self.received_shipment_history = []
        self.departed_shipment_history = []  # ToDo: Track
        self.demand_sum_history = []
        # demand_by_retailer_history / forecast_by_retailer_history are indexed
        # by forecasting CHANNEL (num_retailer entries), not literally by
        # retailer. For a manufacturer a channel is an incoming retailer stream;
        # for a retailer a channel is a product. The legacy names are kept
        # because the forecasting backends and reporting consume them by name.
        self.demand_by_retailer_history = []
        for i in range(self.num_retailer):
            self.demand_by_retailer_history.append([])

        self.forecast_history = []
        self.forecast_by_retailer_history = []
        for i in range(self.num_retailer):
            self.forecast_by_retailer_history.append([])

        # Per-product reporting (one list per product). For single-product
        # agents these mirror the flat lists.
        self.inventory_history_by_product = [[] for _ in range(self.num_products)]
        self.order_history_by_product = [[] for _ in range(self.num_products)]
        self.forecast_history_by_product = [[] for _ in range(self.num_products)]
        self.received_shipment_history_by_product = [[] for _ in range(self.num_products)]

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

        self.variance_ratio = []  # aggregate bullwhip (kept for backward compatibility)
        self.variance_ratio_by_product = [[] for _ in range(self.num_products)]

        # Per-product order computed each replenishment step; consumed by
        # _split_orders to route each product to its supplier(s).
        self.order_by_product = np.zeros(self.num_products, dtype=float)

        ### init order mechanism: How the Order is split among the agents suppliers
        self.__init_splitting_meachanism()
        self._init_product_routing(product_to_manufacturer)

        ### init flags for internal logic
        self.converging = True

        ## others
        # Incoming shipments replenish inventory. One FIFO queue per product,
        # each sized by that product's lead time.
        self.shipment_queues = []
        for p in range(self.num_products):
            lead_p = int(self.lead_time_per_product[p])
            q = queue.Queue(maxsize=lead_p + 1)
            for _ in range(lead_p):
                q.put(0.0)
            self.shipment_queues.append(q)

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

    def _build_lead_times(self, lead_time_matrix) -> np.ndarray:
        """Return per-product lead times of shape (num_products,).

        Supported config shapes (all collapse to the single-product scalar):
          - scalar / size-1            -> broadcast to every product
          - length num_products vector -> one lead time per product
          - 2D (agent x product)       -> row for this agent
        A length num_products vector is only interpreted as per-product when
        num_products > 1, so a per-agent lead-time list at an upstream level is
        not mistaken for per-product values.
        """
        arr = np.asarray(lead_time_matrix)

        if arr.ndim >= 2:
            row = np.asarray(arr[self.id]).reshape(-1)
            if row.size == self.num_products:
                lead = row
            elif row.size == 1:
                lead = np.full(self.num_products, row[0])
            else:
                raise ValueError(
                    f"Agent {self.id} level {self.sc_level}: lead_time row has "
                    f"size {row.size}, expected 1 or num_products={self.num_products}."
                )
        else:
            flat = arr.reshape(-1)
            if self.num_products > 1 and flat.size == self.num_products:
                lead = flat
            elif flat.size == 1:
                lead = np.full(self.num_products, flat[0])
            else:
                # per-agent list (or any other 1D form): use a single value for
                # this agent and broadcast it across products.
                value = flat[self.id] if flat.size > self.id else flat[0]
                lead = np.full(self.num_products, value)

        return np.asarray(lead, dtype=int).reshape(-1)

    def _cfg_per_product(self, key: str) -> np.ndarray:
        """Read a per-agent config entry and expand it to (num_products,).

        The entry cfg[key][self.id] may be a scalar (broadcast to every product)
        or a length num_products list (one value per product). Single-product
        configs therefore work unchanged.
        """
        raw = np.asarray(self.cfg[key][self.id], dtype=float).reshape(-1)
        if raw.size == self.num_products:
            return raw.copy()
        if raw.size == 1:
            return np.full(self.num_products, float(raw[0]))
        raise ValueError(
            f"Agent {self.id} level {self.sc_level}: cfg['{key}'][{self.id}] has "
            f"size {raw.size}, expected 1 or num_products={self.num_products}."
        )

    def __init_replenishment_strat(self) -> None:
        """Sets the replenishment strategy using the inventory capacity, the initial inventory and
        strategy dependend parameters.
        Currently only the Order-Up-To (OUT) Strategy is implemented

        Inventory, capacity and the replenishment policy are held per product.

        Raises:
            NotImplementedError: _description_
        """

        self.current_inv = self._cfg_per_product('init_inv')
        self.inv_capacity = self._cfg_per_product('inv_capacity')

        if self.replenish_strat == "OUT":
            self.R = self._cfg_per_product("R")
            self.risk_factor = self._cfg_per_product('safety_risk_factor')
            self.replenishment = [
                OrderUpTo(
                    R=self.R[p],
                    lead_time=int(self.lead_time_per_product[p]),
                    risk_factor=self.risk_factor[p],
                    inv_cap=self.inv_capacity[p],
                )
                for p in range(self.num_products)
            ]
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

    def _init_product_routing(self, product_to_manufacturer) -> None:
        """Build, per product, which supplier(s) that product's order flows to.

        self.product_routing[p] == (suppliers, shares) where ``suppliers`` are
        global supplier indices and ``shares`` sum to 1.

        The default (product_to_manufacturer is None) routes every product
        equally across all connected suppliers, which reproduces the legacy
        gamma order-split exactly. An explicit map lets a product go 100% (or by
        weighted share) to specific manufacturer(s), e.g. {0: 0, 1: 1, 2: 2}.
        """
        self.product_routing = []
        for p in range(self.num_products):
            entry = None
            if product_to_manufacturer is not None:
                if isinstance(product_to_manufacturer, dict):
                    if p in product_to_manufacturer:
                        entry = product_to_manufacturer[p]
                    elif str(p) in product_to_manufacturer:
                        entry = product_to_manufacturer[str(p)]
                elif p < len(product_to_manufacturer):
                    entry = product_to_manufacturer[p]

            if entry is None:
                suppliers = np.array(self.supplier_index, dtype=int)
                if suppliers.size == 0:
                    shares = np.array([], dtype=float)
                else:
                    shares = np.full(suppliers.size, 1.0 / suppliers.size, dtype=float)
            else:
                suppliers, shares = self._parse_routing_entry(entry, product=p)

            self.product_routing.append((suppliers, shares))

    def _parse_routing_entry(self, entry, product: int):
        """Parse one routing entry into (suppliers, normalized shares).

        Accepted forms:
            int m                 -> 100% to supplier m
            [m0, m1, ...]         -> equal split across the listed suppliers
            {m0: w0, m1: w1, ...} -> weighted split (weights normalized)
        Every referenced supplier must be a connected supplier of this agent.
        """
        if isinstance(entry, dict):
            items = sorted((int(k), float(v)) for k, v in entry.items())
            suppliers = np.array([k for k, _ in items], dtype=int)
            weights = np.array([w for _, w in items], dtype=float)
        elif isinstance(entry, (list, tuple, np.ndarray)):
            suppliers = np.array([int(m) for m in entry], dtype=int)
            weights = np.ones(suppliers.size, dtype=float)
        else:
            suppliers = np.array([int(entry)], dtype=int)
            weights = np.ones(1, dtype=float)

        total = float(np.sum(weights))
        if suppliers.size == 0 or total <= 0.0:
            raise ValueError(
                f"Agent {self.id} level {self.sc_level}: invalid routing entry "
                f"{entry!r} for product {product}."
            )
        shares = weights / total

        connected = set(int(s) for s in self.supplier_index)
        for m in suppliers:
            if int(m) not in connected:
                raise ValueError(
                    f"Agent {self.id} level {self.sc_level}: product {product} is "
                    f"routed to supplier {int(m)}, which is not a connected supplier "
                    f"(connected: {sorted(connected)}). Update the adjacency list or "
                    f"the product_to_manufacturer map."
                )

        return suppliers, shares

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
        """Receive shipments per product and update inventory and reporting lists.

        Each product has its own inventory and its own arrival queue (product p
        ships from its own manufacturer, possibly with a different lead time).
        Flat reporting stores the aggregate; per-product detail is tracked too.
        For a single-product agent this is byte-identical to the scalar version.
        """

        # record inventory BEFORE arrivals (aggregate + per product)
        self.inventory_history.append(float(np.sum(self.current_inv)))
        for p in range(self.num_products):
            self.inventory_history_by_product[p].append(float(self.current_inv[p]))

        total_shipment = 0.0
        for p in range(self.num_products):
            shipment_size = self.shipment_queues[p].get()  # shipment arrives
            shipment_size = self._as_scalar(shipment_size, name="received shipment")

            self.current_inv[p] = float(self.current_inv[p]) + shipment_size

            # ToDo: Track possible waste. Overflow above capacity is lost.
            if self.current_inv[p] > self.inv_capacity[p]:
                self.current_inv[p] = self.inv_capacity[p]

            self.received_shipment_history_by_product[p].append(shipment_size)
            total_shipment += shipment_size

        self.received_shipment_history.append(total_shipment)

    def _sell(self) -> None:
        """Sell the goods, per product.

        1) Track Demand
        2) Update inventory (independently per product)
        3) Track Backlog if necessary

        Each product's inventory serves only the demand channels mapped to that
        product. For a manufacturer (one product, several retailer channels)
        this reduces to the original aggregate behaviour; for a retailer each
        product is settled independently.
        """
        self.current_demand = self._as_vector(self.current_demand, name="current_demand")
        self.current_backlog = self._as_vector(self.current_backlog, name="current_backlog")
        self.current_demand_sum = float(np.sum(self.current_demand))

        self.demand_sum_history.append(self.current_demand_sum)  # add current demand to demand history

        for i in range(self.num_retailer):
            self.demand_by_retailer_history[i].append(float(self.current_demand[i]))

        own_shipments = self._zero_vector()
        new_backlog = self._zero_vector()
        recorded_backlog = self._zero_vector()

        for p in range(self.num_products):
            channels = self._channels_of_product[p]
            incoming_backlog = self.current_backlog[channels]
            required = self.current_demand[channels] + incoming_backlog  # per channel
            total_required = float(np.sum(required))
            inv_p = float(self.current_inv[p])
            new_inv_p = inv_p - total_required

            if new_inv_p < 0:
                # not all demand can be satisfied: ship proportionally, backlog rest
                if total_required <= 0:
                    proportions = np.zeros(channels.size, dtype=float)
                else:
                    proportions = required / total_required
                available_inventory = max(inv_p, 0.0)
                ship = np.trunc(available_inventory * proportions)
                self.current_inv[p] = inv_p - float(np.sum(ship))
                resulting_backlog = required - ship
                own_shipments[channels] = ship
                new_backlog[channels] = resulting_backlog
                recorded_backlog[channels] = np.abs(resulting_backlog)
            else:
                own_shipments[channels] = required
                self.current_inv[p] = new_inv_p
                # channels of a fully-served product carry no backlog forward;
                # record the (now cleared) incoming backlog, matching the legacy
                # single-product bookkeeping.
                recorded_backlog[channels] = np.abs(incoming_backlog)

        self.current_backlog = new_backlog
        self.back_log_list.append(recorded_backlog)

        self.own_shipments = self._as_vector(own_shipments, name="own_shipments")
        self.departed_shipment_history.append(self.own_shipments.copy())

    def _compute_orders(self, d_est) -> None:
        """Turn a per-channel forecast into per-product orders and record them.

        d_est is one forecast per demand channel. Each product's order is
        computed from the forecast(s) of the channels feeding it and that
        product's own inventory. For a retailer channel==product, so this is a
        strict per-product order; for a manufacturer all channels feed the single
        product, so the forecasts are summed (the original aggregate behaviour).
        """
        d_est = self._as_vector(d_est, name="forecast d_est")

        for i in range(self.num_retailer):
            self.forecast_by_retailer_history[i].append(float(d_est[i]))

        # aggregate forecast, kept for backward-compatible flat reporting
        self.forecast_history.append(float(np.sum(d_est)))

        orders = np.zeros(self.num_products, dtype=float)
        for p in range(self.num_products):
            channels = self._channels_of_product[p]
            d_est_p = float(np.sum(d_est[channels]))
            order_p = self.replenishment[p].compute_order_sum(
                d_est=d_est_p,
                currnet_inv=float(self.current_inv[p]),
            )
            orders[p] = order_p
            self.order_history_by_product[p].append(order_p)
            self.forecast_history_by_product[p].append(d_est_p)

        self.order_by_product = orders
        # aggregate order, kept for backward-compatible flat reporting / bullwhip
        self.order_history.append(float(np.sum(orders)))

    def _replenish_inv(self) -> None:
        """Replenish the Inventory by Forecasting Demand and Calculating the Order Size
        """
        data = self.__build_prediction_feature()  # get data for prediction
        d_est = self.forecasting_model.predict(data)  # forecast the demand
        self._compute_orders(d_est)

    def _replenish_inv_multichannel(self, predictions) -> None:
        """Replenish the Inventory using externally supplied (collaborative) forecasts.
        """
        # __build_prediction_feature is kept for parity with the original call
        # order even though the prediction is passed in directly.
        _ = self.__build_prediction_feature()
        self._compute_orders(predictions)

    def _split_orders(self) -> list:
        """Route this agent's orders to its suppliers.

        Retailers (sc_level 0) route each product's order to that product's
        supplier(s) via the product routing map. With the default routing (equal
        split across all connected suppliers) and a single product this
        reproduces the legacy gamma split exactly. Upstream agents keep the
        legacy aggregate gamma split.

        Returns:
            list: order size for each supplier of the next supply chain level
        """
        if self.sc_level == 0:
            return self._split_orders_by_product()

        # legacy aggregate split for upstream (single-product) agents
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

    def _split_orders_by_product(self) -> list:
        """Route each product's order to its configured supplier(s).

        Each product p distributes order_by_product[p] across product_routing[p]
        using the same round-all-but-last-supplier scheme as the legacy split,
        so the single-product default is byte-identical. Contributions from
        every product accumulate into one order per supplier.
        """
        demand_list = np.zeros(self.supplier_num, dtype=float)

        for p in range(self.num_products):
            suppliers, shares = self.product_routing[p]
            order_p = float(self.order_by_product[p])

            if suppliers.size == 0:
                continue

            dist = order_p * np.asarray(shares, dtype=float)
            d = 0
            for i in range(len(dist) - 1):
                dist[i] = np.round(dist[i])
                d = np.round(dist[i]) + d
            dist[-1] = order_p - d

            for i, ind_ in enumerate(suppliers):
                demand_list[int(ind_)] += dist[i]

        demand_list = demand_list.tolist()
        self.order_per_supplier_history.append(demand_list)

        return demand_list

    def act(self, demand_t: int, sum_received_shipments_t: int) -> list:
        """Execute one simulation step using the agent's forecasting model.

            The method records the current demand, receives shipments from the
            agent's internal shipment queues, satisfies downstream demand,
            computes replenishment orders, and routes those orders to upstream
            suppliers.

            Args:
                demand_t:
                    Demand for the current time step. The value is normalized to
                    one entry per downstream demand channel.
                sum_received_shipments_t:
                    Deprecated compatibility parameter. It is currently unused;
                    incoming shipments are read from the agent's internal
                    per-product shipment queues.

            Returns:
                tuple[list[float], numpy.ndarray]:
                    A tuple containing:

                    1. the order quantity addressed to each upstream supplier;
                    2. the outgoing shipment quantity for each downstream demand
                    channel.
            """
        self.current_demand = self._as_vector(demand_t, name="demand_t")
        self.current_demand_sum = float(np.sum(self.current_demand))

        self._receive_shipment()
        self._sell()
        self._replenish_inv()

        self.own_shipments = self._as_vector(self.own_shipments, name="own_shipments")
        return self._split_orders(), self.own_shipments.copy()  # order for own supplier, and shipments to own retailer

    def act_multichannel(self, demand_t: int, sum_received_shipments_t: int, predictions) -> list:
        """Execute one simulation step using externally computed forecasts.

            This follows the same inventory, sales, and shipment workflow as
            ``act``, but uses the supplied per-channel predictions when computing
            replenishment orders.

            Args:
                demand_t:
                    Demand for the current time step. The value is normalized to
                    one entry per downstream demand channel.
                sum_received_shipments_t:
                    Deprecated compatibility parameter. It is currently unused;
                    incoming shipments are read from the agent's internal
                    per-product shipment queues.
                predictions:
                    Forecast value for each demand channel, supplied by the
                    collaborative forecasting backend.

            Returns:
                tuple[list[float], numpy.ndarray]:
                    A tuple containing the orders per upstream supplier and the
                    outgoing shipments per downstream demand channel.
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

    def set_shipment(self, shipment) -> None:
        """Enqueue an incoming shipment into the per-product arrival queues.

        A per-product vector (length num_products) routes each entry to its
        product queue. A scalar is only valid for a single-product agent (e.g. a
        manufacturer replenishing its one product) and matches the legacy path.
        """
        arr = np.asarray(shipment, dtype=float).reshape(-1)

        if arr.size == self.num_products:
            for p in range(self.num_products):
                self.shipment_queues[p].put(float(arr[p]))
        elif arr.size == 1:
            if self.num_products != 1:
                raise ValueError(
                    f"Agent {self.id} level {self.sc_level}: set_shipment received a "
                    f"scalar but num_products={self.num_products}. Provide one value "
                    "per product."
                )
            self.shipment_queues[0].put(float(arr[0]))
        else:
            raise ValueError(
                f"Agent {self.id} level {self.sc_level}: set_shipment received shape "
                f"{np.shape(shipment)}, expected num_products={self.num_products}."
            )

    def _product_demand_series(self, p: int) -> np.ndarray:
        """Aggregate demand history for product p (sum over its channels)."""
        channels = self._channels_of_product[p]
        series = np.zeros(len(self.demand_sum_history), dtype=float)
        for c in channels:
            series += np.asarray(self.demand_by_retailer_history[c], dtype=float)
        return series

    def get_variance_ratio(self, last_t: int) -> float:

        var_demand = np.var(self.demand_sum_history[-last_t:])
        var_orders = np.var(self.order_history[-last_t:])

        return np.round((var_orders / var_demand), 2)

    def get_variance_ratio_by_product(self, last_t: int) -> list:
        """Per-product bullwhip (order variance / demand variance).

        Measuring per product avoids hiding amplification behind the aggregate
        for a multi-product retailer.
        """
        ratios = []
        for p in range(self.num_products):
            demand_series = self._product_demand_series(p)[-last_t:]
            order_series = np.asarray(self.order_history_by_product[p][-last_t:], dtype=float)
            var_demand = np.var(demand_series)
            var_orders = np.var(order_series)
            ratios.append(np.round((var_orders / var_demand), 2))
        return ratios

    def set_variance_ratio(self, last_t: int) -> None:

        self.variance_ratio.append(self.get_variance_ratio(last_t))
        per_product = self.get_variance_ratio_by_product(last_t)
        for p in range(self.num_products):
            self.variance_ratio_by_product[p].append(per_product[p])
