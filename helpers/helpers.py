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

def _create_market_from_cfg(
    cfg: dict,
    T: int,
    agents_per_level,
):
    """
    Create either the original artificial market or a real-data market.

    Backward compatibility:
        market.data_scource: null  -> MarketArtifical, exactly like before
        market.data_scource: path  -> MarketDataSource using Excel/CSV demand
    """
    market_cfg = cfg["market"]
    data_source = _get_market_data_source(market_cfg)

    market_demand_split = market_cfg["demand_split"]
    retailer_num = agents_per_level[0]

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
            )
            agent_list.append(agent)

        sc_agent_list.append(agent_list)

    supply_chain = Supply_Chain(sc_adjacency, sc_lead_time)

    return simulation, market, supply_chain, sc_agent_list


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
