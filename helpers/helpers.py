import logging
import yaml
import json
import subprocess
import re

import numpy as np
import pandas as pd
import torch

from copy import deepcopy
from datetime import datetime
from pathlib import Path

from Simulation_Component.market import *
from Simulation_Component.agent import *
from Simulation_Component.supply_chain import *
from Simulation_Component.simulation import Simulation

logger = logging.getLogger("logger")


# =============================================================================
# Config loading
# =============================================================================

def load_config(path: Path):
    """Load the config for the simulation."""
    with open(path, "r") as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)

    return cfg


# =============================================================================
# Real-data demand loading
# =============================================================================

def _normalize_column_name(value) -> str:
    """
    Normalize a config/Excel/CSV column name to make matching robust.

    Example:
        "Week Ending Date" -> "week_ending_date"
        "Final Butter Sales" -> "final_butter_sales"
    """
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_column_name(col) for col in df.columns]
    return df


def _is_missing_data_source(data_source) -> bool:
    """
    Treat None, empty string, 'null', 'none', and NaN as missing.
    This keeps market.data_scource: null fully backward-compatible.
    """
    if data_source is None:
        return True

    if isinstance(data_source, float) and np.isnan(data_source):
        return True

    if isinstance(data_source, str):
        return data_source.strip().lower() in {"", "none", "null", "nan"}

    return False


def _get_market_data_source(market_cfg: dict):
    """
    Keep the original misspelled key data_scource, but also allow data_source.
    """
    if "data_scource" in market_cfg:
        return market_cfg.get("data_scource")

    return market_cfg.get("data_source")


def _clean_numeric_series(series: pd.Series) -> np.ndarray:
    """
    Convert values such as '1,234', '$1,234', and blanks to clean floats.
    Non-numeric values are dropped.
    """
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaT": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )


def _read_market_data_file(data_path: Path, market_cfg: dict) -> pd.DataFrame:
    """
    Read .xlsx/.xls/.csv files. Excel sheet can be configured with:
        market.sheet_name: 0
    """
    suffix = data_path.suffix.lower()
    sheet_name = market_cfg.get("sheet_name", 0)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(data_path, sheet_name=sheet_name)

    if suffix == ".csv":
        return pd.read_csv(data_path)

    raise ValueError(
        f"Unsupported market data file type: {data_path.suffix}. "
        "Use .xlsx, .xls, or .csv."
    )


def _infer_demand_column(df: pd.DataFrame, configured_column=None):
    """
    Infer the demand/sales column if market.demand_column is not set.

    Priority:
      1. Configured demand_column if it exists.
      2. Exact 'sales' column.
      3. Columns containing sales/demand/quantity/volume/pounds.
      4. A single usable numeric column after excluding date/meta columns.
      5. None, which triggers wide-row fallback.
    """
    if configured_column is not None:
        configured_column = _normalize_column_name(configured_column)
        if configured_column in df.columns:
            return configured_column

    if "sales" in df.columns:
        return "sales"

    include_tokens = ["sales", "demand", "quantity", "volume", "pounds", "lbs"]
    exclude_tokens = [
        "price",
        "moisture",
        "date",
        "week",
        "year",
        "month",
        "product",
        "section",
        "source",
        "report",
        "status",
    ]

    candidate_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(token in col_lower for token in include_tokens) and not any(
            token in col_lower for token in exclude_tokens
        ):
            candidate_columns.append(col)

    if candidate_columns:
        # Use the candidate with the most numeric values.
        scored = []
        for col in candidate_columns:
            scored.append((len(_clean_numeric_series(df[col])), col))

        scored.sort(reverse=True)
        if scored[0][0] > 0:
            return scored[0][1]

    # Fallback: one numeric-like non-meta column.
    meta_tokens = ["date", "week", "year", "month", "product", "section", "source", "report"]
    numeric_like = []
    for col in df.columns:
        if any(token in col.lower() for token in meta_tokens):
            continue

        values = _clean_numeric_series(df[col])
        if len(values) > 0:
            numeric_like.append((len(values), col))

    if len(numeric_like) == 1:
        return numeric_like[0][1]

    if len(numeric_like) > 1:
        numeric_like.sort(reverse=True)
        return numeric_like[0][1]

    return None


def load_market_demand_from_file(market_cfg: dict) -> np.ndarray:
    """
    Load real demand values from one Excel or CSV file.

    Config fields:
        market.data_scource:       path to .xlsx/.xls/.csv
        market.data_source:        alternative spelling, optional
        market.dataset_name:       optional, only for reporting/debugging
        market.demand_column:      optional; if missing, inferred
        market.date_column:        optional; used only for sorting
        market.sheet_name:         optional for Excel; default 0
        market.demand_scale_divisor: optional; default 1000 to preserve old behavior
        market.demand_scale_multiplier: optional; default 1

    The old implementation did:
        data = values / 1000
    Therefore this loader defaults to demand_scale_divisor=1000.
    Set demand_scale_divisor: 1 in the config if you want raw values.
    """
    data_source = _get_market_data_source(market_cfg)

    if _is_missing_data_source(data_source):
        raise ValueError("market.data_scource is null; no real-data file was provided.")

    data_path = Path(str(data_source)).expanduser()

    if not data_path.exists():
        raise FileNotFoundError(f"Market demand file not found: {data_path}")

    df = _read_market_data_file(data_path, market_cfg)
    df = _normalize_dataframe_columns(df)

    date_column = market_cfg.get("date_column")
    if date_column is not None:
        date_column = _normalize_column_name(date_column)
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df = df.sort_values(date_column)

    demand_column = _infer_demand_column(
        df=df,
        configured_column=market_cfg.get("demand_column"),
    )

    if demand_column is not None:
        demand = _clean_numeric_series(df[demand_column])
    else:
        # Backward-compatible fallback for older wide Excel sheets where the
        # first row contains the whole time series across columns.
        first_row = pd.Series(df.iloc[0].values)
        demand = _clean_numeric_series(first_row)

    if len(demand) == 0:
        raise ValueError(
            f"No numeric demand values found in {data_path}. "
            f"Set market.demand_column explicitly. Available columns: {list(df.columns)}"
        )

    demand_scale_divisor = float(market_cfg.get("demand_scale_divisor", 1000))
    demand_scale_multiplier = float(market_cfg.get("demand_scale_multiplier", 1))

    if demand_scale_divisor == 0:
        raise ValueError("market.demand_scale_divisor must not be 0.")

    demand = (demand / demand_scale_divisor) * demand_scale_multiplier

    return np.round(demand, 0)


# =============================================================================
# Simulation object creation
# =============================================================================

def _create_market_factor_model(market_cfg, T, retailer_num, num_products):
    """Construct the multi-product MarketFactorModel.

    The class is provided separately (market_factor_model.py / re-exported from
    Simulation_Component.market). It must emit R*P demand streams in order
    s = retailer*P + product via split_demand_on_time(t). This is the single
    integration point: if the class's constructor signature differs, adjust the
    keyword arguments here.
    """
    try:
        # Prefer an explicit module; fall back to the market package export.
        try:
            from Simulation_Component.market_factor_model import MarketFactorModel
        except ImportError:
            from Simulation_Component.market import MarketFactorModel  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "num_products > 1 requires MarketFactorModel, which was not found. "
            "Add Simulation_Component/market_factor_model.py (or export "
            "MarketFactorModel from Simulation_Component.market)."
        ) from exc

    random_walk = market_cfg["random_walk"]
    # market.demand_split must be an R*P vector (stream order s = retailer*P + p)
    # summing to 1.
    market_demand_split = market_cfg["demand_split"]

    # Optional factor-model knobs (safe defaults reproduce a single shared
    # signal). These are the levers the collaborative-forecasting study sweeps:
    #   lam  in [0,1] : fraction of each stream's fluctuation from the shared factor
    #   tau  >= 0     : lead-lag delay between consecutive products
    factor_cfg = market_cfg.get("factor_model", {}) if isinstance(market_cfg.get("factor_model"), dict) else {}

    def _fm(key, default):
        if key in factor_cfg:
            return factor_cfg[key]
        return market_cfg.get(key, default)

    kwargs = dict(
        product_num=num_products,
        lam=float(_fm("lam", 1.0)),
        tau=int(_fm("tau", 0)),
        idio_noise_scale=float(_fm("idio_noise_scale", 1.0)),
        shock_prob=float(_fm("shock_prob", 0.0)),
        shock_mag=float(_fm("shock_mag", 0.0)),
    )
    idio_freqs = _fm("idio_freqs", None)
    idio_phases = _fm("idio_phases", None)
    seed = _fm("seed", None)
    if idio_freqs is not None:
        kwargs["idio_freqs"] = idio_freqs
    if idio_phases is not None:
        kwargs["idio_phases"] = idio_phases
    if seed is not None:
        kwargs["seed"] = seed

    return MarketFactorModel(
        T,
        market_cfg["primary_demand"],
        market_cfg["trend_magnitude"],
        market_cfg["seasonality_magnitude"],
        market_cfg["seasonality_frequncy"],
        random_walk["mean"],
        random_walk["variance"],
        retailer_num,
        market_demand_split,
        **kwargs,
    )


def _create_market_data_source_multi_product(market_cfg, T, retailer_num, num_products):
    """Build MarketDataSourceMultiProduct from one real data series per product.

    market.data_scource must be a list with one entry per product. Each entry is
    either a file path (string) or a dict of per-product overrides
    (data_scource/data_source, demand_column, demand_scale_divisor,
    demand_scale_multiplier, dataset_name). Each series is loaded with the same
    loader used for the single-product real-data path.

    market.demand_split is the R*P retailer-share vector (order s = retailer*P +
    product), where each product's R shares sum to 1 (validated in the market).
    """
    entries = _get_market_data_source(market_cfg)
    if not isinstance(entries, (list, tuple)):
        raise ValueError(
            "num_products > 1 with real data requires market.data_scource to be a "
            "list with one entry per product (path or per-product dict). "
            f"Got {type(entries).__name__}."
        )
    if len(entries) != num_products:
        raise ValueError(
            f"market.data_scource has {len(entries)} entries but num_products="
            f"{num_products}. Provide exactly one demand series per product."
        )

    product_data = []
    for p, entry in enumerate(entries):
        per_cfg = dict(market_cfg)
        if isinstance(entry, dict):
            per_cfg.update(entry)
            # normalise the (optionally correctly-spelled) source key
            if "data_source" in entry and "data_scource" not in entry:
                per_cfg["data_scource"] = entry["data_source"]
        else:
            per_cfg["data_scource"] = entry

        data = load_market_demand_from_file(per_cfg)
        if len(data) < T:
            raise ValueError(
                f"Product {p} demand series is too short: length {len(data)} < "
                f"required T={T}."
            )
        product_data.append(np.asarray(data, dtype=float))

    return MarketDataSourceMultiProduct(
        product_data=product_data,
        retailer_num=retailer_num,
        num_products=num_products,
        market_demand_split=market_cfg["demand_split"],
    )


def _create_market_from_cfg(
    cfg: dict,
    T: int,
    agents_per_level,
    num_products: int = 1,
):
    """
    Create either the original artificial market or a real-data market.

    Backward compatibility:
        market.data_scource: null  -> MarketArtifical, exactly like before
        market.data_scource: path  -> MarketDataSource using Excel/CSV demand

    Multi-product (num_products > 1) uses MarketFactorModel, which emits one
    demand stream per (retailer, product) pair in order s = retailer*P + product.
    """
    market_cfg = cfg["market"]
    data_source = _get_market_data_source(market_cfg)

    market_demand_split = market_cfg["demand_split"]
    retailer_num = agents_per_level[0]

    if num_products > 1:
        # Real per-product data if a source is given, otherwise the synthetic
        # factor model.
        if not _is_missing_data_source(data_source):
            return _create_market_data_source_multi_product(
                market_cfg=market_cfg,
                T=T,
                retailer_num=retailer_num,
                num_products=num_products,
            )
        return _create_market_factor_model(
            market_cfg=market_cfg,
            T=T,
            retailer_num=retailer_num,
            num_products=num_products,
        )

    if _is_missing_data_source(data_source):
        random_walk = market_cfg["random_walk"]

        return MarketArtifical(
            sim_time=T,
            primary_demand=market_cfg["primary_demand"],
            trend_mag=market_cfg["trend_magnitude"],
            seasonality_mag=market_cfg["seasonality_magnitude"],
            seasonality_freq=market_cfg["seasonality_frequncy"],
            random_walk_mean=random_walk["mean"],
            random_walk_var=random_walk["variance"],
            retailer_num=retailer_num,
            market_demand_split=market_demand_split,
        )

    data = load_market_demand_from_file(market_cfg)

    # =========================
    # DEBUG REAL DATA
    # =========================
    print("=" * 80, flush=True)
    print("REAL DATA DEBUG", flush=True)
    print("data_source:", data_source, flush=True)
    print("dataset_name:", market_cfg.get("dataset_name"), flush=True)
    print("demand_column:", market_cfg.get("demand_column"), flush=True)
    print("type:", type(data), flush=True)
    print("shape:", getattr(data, "shape", None), flush=True)
    print("dtype:", getattr(data, "dtype", None), flush=True)
    print("len:", len(data), flush=True)
    print("T required:", T, flush=True)
    print("first 10:", data[:10], flush=True)
    print("last 10:", data[-10:], flush=True)
    print("min:", np.min(data), flush=True)
    print("max:", np.max(data), flush=True)
    print("nan count:", np.isnan(data).sum(), flush=True)
    print("inf count:", np.isinf(data).sum(), flush=True)
    print("=" * 80, flush=True)
    # =========================

    if len(data) < T:
        raise ValueError(
            f"Simulation time is too long for the given dataset. "
            f"Required T={T}, but loaded demand length is {len(data)}. "
            f"Reduce convergence_time + simulation_time + testing_time, "
            f"or use a longer data file."
        )

    training_time = int(cfg["sim"]["training_time"])
    if len(data) < training_time:
        raise ValueError(
            f"Training time is too long for the given dataset. "
            f"training_time={training_time}, but loaded demand length is {len(data)}."
        )

    return MarketDataSource(
        data=data,
        retailer_num=retailer_num,
        market_demand_split=market_demand_split,
    )


def _build_simulation_objects_from_cfg(cfg: dict):
    """
    Shared implementation for:
      - init_simulaltion(path)
      - reset_simulaltion(path)
      - reset_simulaltion_from_dict(cfg)

    This removes the duplicated market-creation code and fixes the old typo:
        marekt = MarketDataSource(...)
    """
    # CONFIG
    sim_time = cfg["sim"]["simulation_time"]
    conv_time = cfg["sim"]["convergence_time"]
    sim_runs = cfg["sim"]["simulation_runs"]
    training_time = cfg["sim"]["training_time"]
    testing_time = cfg["sim"]["testing_time"]
    training_type = cfg["sim"]["training_type"]
    train_size = cfg["sim"]["train_size"]
    val_size = cfg["sim"]["val_size"]

    T = conv_time + sim_time + testing_time
    epochs = cfg["sim"]["epochs"]
    learning_rate = cfg["sim"]["learning_rate"]
    momentum = cfg["sim"]["momentum"]
    batch_size = cfg["sim"]["batch_size"]
    sequence_length = cfg["sim"]["sequence_length"]

    sc_all = cfg["supply_chain"]
    agents_per_level = sc_all["agents_per_level"]
    sc_levels = sc_all["sc_levels"]

    # Multi-product settings. num_products defaults to 1, which collapses to the
    # original single-product semantics. product_to_manufacturer is an optional
    # routing map (product index -> supplier(s)); None means "equal split across
    # all connected suppliers", reproducing the legacy gamma split.
    num_products = int(
        sc_all.get("num_products", cfg.get("market", {}).get("num_products", 1))
    )
    product_to_manufacturer = sc_all.get("product_to_manufacturer", None)

    simulation = Simulation(
        T=T,
        sim_runs=sim_runs,
        sim_time=sim_time,
        conv_time=conv_time,
        retraining_time=training_time,
        testing_time=testing_time,
        training_type=training_type,
        train_size=train_size,
        val_size=val_size,
    )

    market = _create_market_from_cfg(
        cfg=cfg,
        T=T,
        agents_per_level=agents_per_level,
        num_products=num_products,
    )

    sc_agent_list = []
    sc_adjacency = []
    sc_lead_time = []

    # Init Agents
    for i, level in enumerate(sc_levels):
        num_agents = agents_per_level[i]
        sc = sc_levels[level]

        adjaceny_matrix = np.array(sc["adjaceny_list"])
        lead_time_matrix = np.array(sc["lead_time"])

        sc_adjacency.append(adjaceny_matrix)
        sc_lead_time.append(lead_time_matrix)

        agent_list = []
        for j in range(num_agents):
            agent = Agent(
                id=j,
                sc_level=i,
                adjacency_matrix=adjaceny_matrix,
                lead_time_matrix=lead_time_matrix,
                training_time=training_time,
                cfg=sc,
                cfg_all=sc_all,
                sequence_length=sequence_length,
                eopchs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                momentum=momentum,
                num_products=num_products,
                # Only retailers (level 0) use the product routing map.
                product_to_manufacturer=product_to_manufacturer if i == 0 else None,
            )
            agent_list.append(agent)

        sc_agent_list.append(agent_list)

    supply_chain = Supply_Chain(sc_adjacency, sc_lead_time)

    # Expose multi-product routing to the runner. product_to_suppliers[p] is the
    # list of manufacturer indices that produce product p, derived from the
    # retailers' routing so it always matches what the retailers actually do.
    supply_chain.num_products = num_products
    supply_chain.product_to_suppliers = _derive_product_to_suppliers(sc_agent_list)
    _assert_routing_valid(sc_agent_list, supply_chain.product_to_suppliers)

    return simulation, market, supply_chain, sc_agent_list


def _assert_routing_valid(sc_agent_list, product_to_suppliers) -> None:
    """Fail loudly unless each manufacturer receives exactly one product's streams.

    A silent mis-routing (a manufacturer fed by two different products, or a
    manufacturer's channel count not matching the retailers that actually route
    to it) would invalidate every experiment, so it is checked here rather than
    left to surface as a subtle numeric error.
    """
    if len(sc_agent_list) < 2:
        return

    retailers = sc_agent_list[0]
    manufacturers = sc_agent_list[1]

    # inverse map: manufacturer index -> set of products routed to it
    manufacturer_products = {}
    for p, suppliers in enumerate(product_to_suppliers):
        for m in suppliers:
            manufacturer_products.setdefault(int(m), set()).add(p)

    for m, products in manufacturer_products.items():
        if len(products) != 1:
            raise ValueError(
                f"Manufacturer {m} would receive order streams for products "
                f"{sorted(products)}. Each manufacturer must produce exactly one "
                "product; check product_to_manufacturer."
            )
        product = next(iter(products))
        # every retailer routing this product to m contributes one channel
        senders = sum(
            1 for r in retailers
            if m in {int(s) for s in r.product_routing[product][0]}
        )
        if manufacturers[m].num_retailer != senders:
            raise ValueError(
                f"Manufacturer {m} has num_retailer={manufacturers[m].num_retailer} "
                f"channels but {senders} retailer(s) route product {product} to it. "
                "The level-0 adjacency and the product routing are inconsistent."
            )


def _derive_product_to_suppliers(sc_agent_list) -> list:
    """Build product_index -> [manufacturer indices] from the retailers' routing.

    Asserts every retailer routes each product to the same set of suppliers, so a
    single global mapping is well defined (required by the per-product shipment
    reassembly in the runner). Fails loudly otherwise.
    """
    retailers = sc_agent_list[0]
    if not retailers:
        return []

    num_products = retailers[0].num_products
    product_to_suppliers = []
    for p in range(num_products):
        suppliers = sorted(int(s) for s in retailers[0].product_routing[p][0])
        for r in retailers[1:]:
            other = sorted(int(s) for s in r.product_routing[p][0])
            if other != suppliers:
                raise ValueError(
                    f"Retailer {r.id} routes product {p} to suppliers {other}, but "
                    f"retailer {retailers[0].id} routes it to {suppliers}. "
                    "Per-retailer routing differences are not supported; use a "
                    "consistent product_to_manufacturer map."
                )
        product_to_suppliers.append(suppliers)
    return product_to_suppliers


def init_simulaltion(path: Path) -> list[Simulation, Market, Supply_Chain, list, dict]:
    """
    Initiate all objects needed for simulations from a config path.

    Returns:
        simulation, market, supply_chain, sc_agent_list, cfg
    """
    cfg = load_config(path)
    simulation, market, supply_chain, sc_agent_list = _build_simulation_objects_from_cfg(cfg)
    return simulation, market, supply_chain, sc_agent_list, cfg


def reset_simulaltion(path: Path) -> list[Simulation, Market, Supply_Chain, list, dict]:
    """
    Reset all objects needed for simulations from a config path.

    Returns:
        simulation, market, supply_chain, sc_agent_list, cfg
    """
    cfg = load_config(path)
    simulation, market, supply_chain, sc_agent_list = _build_simulation_objects_from_cfg(cfg)
    return simulation, market, supply_chain, sc_agent_list, cfg


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset.

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def reset_simulaltion_from_dict(cfg: dict) -> list[Simulation, Market, Supply_Chain, list, dict]:
    """
    Initiate all objects needed for simulations from an already merged config dict.

    Important:
        main.py currently expects this function to return four values:
            simulation, market, supply_chain, sc_agent_list

        Therefore this function intentionally does NOT return cfg.
    """
    return _build_simulation_objects_from_cfg(cfg)


# =============================================================================
# GPU helpers
# =============================================================================

def select_gpu():
    if torch.cuda.is_available():
        return select_least_used_gpu()

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_gpu_usage():
    # Run nvidia-smi command to get GPU usage
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
    )

    # Decode result and split by newlines to get usage for each GPU
    usage = result.stdout.decode("utf-8").split("\n")

    # Remove empty strings and convert to integers
    usage = [int(x) for x in usage if x]

    return usage


def select_least_used_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your setup.")

    usage = get_gpu_usage()
    if not usage:
        raise RuntimeError("No GPU usage information available.")

    # Select the GPU with the lowest usage
    least_used_gpu = usage.index(min(usage))

    # Set this GPU as the default device
    torch.cuda.set_device(least_used_gpu)
    device = torch.cuda.current_device()
    logger.info(f"Selected GPU {least_used_gpu} with usage {usage[least_used_gpu]}%")

    return device


# =============================================================================
# Experiment config helpers
# =============================================================================

def deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value

    return base


def build_experiment_configs(config_path: Path):
    with open(config_path, "r") as file:
        base_cfg = yaml.safe_load(file)

    experiments = base_cfg.pop("experiments", None)

    if not experiments:
        return [("default", base_cfg)]

    experiment_configs = []

    for experiment in experiments:
        name = experiment["name"]
        overrides = experiment.get("overrides", {})

        cfg = deepcopy(base_cfg)
        cfg = deep_update(cfg, overrides)

        # Optional, but useful for reporting/debugging
        cfg.setdefault("meta", {})
        cfg["meta"]["experiment_name"] = name

        experiment_configs.append((name, cfg))

    return experiment_configs
