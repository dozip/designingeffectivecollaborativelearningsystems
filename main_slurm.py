import os
import argparse

# Set these before importing numpy / torch-heavy modules.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import json
import random
import shutil
import logging
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from collections import deque


# =============================================================================
# GPU allocation mode
# =============================================================================
#
# The parent scheduler configures these values from CLI arguments in run().
# Child processes do not need to rediscover the allocation: the parent narrows
# CUDA_VISIBLE_DEVICES before each child is spawned, so every GPU worker sees
# exactly one device as cuda:0.

GPU_ALLOCATION_MODE = "fixed"
FIXED_GPU_IDS: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7"]
GPU_SLOTS: list[dict] = []

import numpy as np
import psutil
import torch
import yaml

from helpers.helpers import (
    build_experiment_configs,
    reset_simulaltion_from_dict,
)


# =============================================================================
# Logging
# =============================================================================
#
# Keep logger quiet so the terminal dashboard can refresh in-place.
# Warnings/errors still appear.

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.WARNING,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger("logger")
logger.setLevel(logging.WARNING)


# =============================================================================
# Live logging helpers
# =============================================================================
#
# The parent scheduler keeps a small live event log and a machine-readable
# failure index. Each child process writes its own live run log. This keeps
# tracebacks readable and avoids many experiment processes appending to one
# shared output file at the same time.

RUN_LIVE_LOG_FILENAME = "run_live.log"
SCHEDULER_LIVE_LOG_FILENAME = "scheduler_live.log"
FAILURES_JSONL_FILENAME = "failures.jsonl"


def utc_timestamp_for_log() -> str:
    """
    Timestamp for log lines. Kept as local wall-clock time to match folder names
    and terminal output.
    """
    return datetime.now().isoformat(timespec="seconds")


def build_run_reporting_path(
    reporting_path: Path,
    timestamp: str,
    experiment_name: str,
    cfg: dict,
    run_id: int,
) -> Path:
    """
    Return the final run folder used by reporting and live logs.

    Final layout:
        Reporting/<timestamp>/<experiment_folder>/run_<run_id>/
    """
    parent_reporting_path = Path(reporting_path) / str(timestamp)
    experiment_folder_name = build_experiment_folder_name(
        experiment_name=experiment_name,
        cfg=cfg,
    )

    return parent_reporting_path / experiment_folder_name / f"run_{run_id}"


def build_run_live_log_path(
    reporting_path: Path,
    timestamp: str,
    experiment_name: str,
    cfg: dict,
    run_id: int,
) -> Path:
    """
    Return the per-run live log path.
    """
    return (
        build_run_reporting_path(
            reporting_path=reporting_path,
            timestamp=timestamp,
            experiment_name=experiment_name,
            cfg=cfg,
            run_id=run_id,
        )
        / RUN_LIVE_LOG_FILENAME
    )


def append_text_line(path: Path, line: str):
    """
    Append one line to a text file and flush immediately.

    This is used only by the parent process for scheduler_live.log and
    failures.jsonl, so no cross-process lock is needed here. Per-run output is
    written by exactly one child process per run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8", buffering=1) as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()


def append_scheduler_log(reporting_path: Path, timestamp: str, message: str):
    """
    Append a parent-scheduler event to Reporting/<timestamp>/scheduler_live.log.

    This file is safe to inspect while the scheduler is running, for example:
        tail -f Reporting/<timestamp>/scheduler_live.log
    """
    if not message:
        return

    try:
        append_text_line(
            Path(reporting_path) / str(timestamp) / SCHEDULER_LIVE_LOG_FILENAME,
            f"[{utc_timestamp_for_log()}] {message}",
        )
    except Exception:
        # Logging should never crash the scheduler.
        pass


def append_jsonl(path: Path, record: dict):
    """
    Append one JSON object as one line and flush immediately.
    """
    try:
        append_text_line(
            path,
            json.dumps(record, ensure_ascii=False, default=str),
        )
    except Exception:
        # Failure bookkeeping should never mask the original scheduler state.
        pass


def append_failure_record(
    job: dict,
    pid: int,
    exitcode: int | None,
    gpu_id: int | None,
):
    """
    Append one failed-run record to Reporting/<timestamp>/failures.jsonl.

    The full traceback/output lives in the per-run run_live.log named by the
    run_log field.
    """
    reporting_path = Path(job["reporting_path"])
    timestamp = str(job["timestamp"])
    run_id = int(job["run_id"])
    run_id_label = job.get("run_id_label", f"run_{run_id}")

    run_log_path = build_run_live_log_path(
        reporting_path=reporting_path,
        timestamp=timestamp,
        experiment_name=job["experiment_name"],
        cfg=job["cfg"],
        run_id=run_id,
    )

    failures_path = reporting_path / timestamp / FAILURES_JSONL_FILENAME

    append_jsonl(
        failures_path,
        {
            "time": utc_timestamp_for_log(),
            "pid": pid,
            "exitcode": exitcode,
            "training_type": job.get("training_type"),
            "experiment_name": job["experiment_name"],
            "run_id": run_id,
            "run_id_label": run_id_label,
            "freq": job.get("freq", "NA"),
            "noise": job.get("noise", "NA"),
            "magnitude": job.get("magnitude", "NA"),
            "lead_time": job.get("lead_time", "NA"),
            "gpu_id": gpu_id,
            "run_log": str(run_log_path),
        },
    )


def get_process_entry_context(args, kwargs) -> dict:
    """
    Extract named values from run_single_experiment_process_entry arguments.

    The scheduler currently passes positional args, but supporting kwargs makes
    this wrapper safer if the entry point is reused later.
    """
    names = [
        "experiment_id",
        "total_experiments",
        "experiment_name",
        "cfg",
        "run_id",
        "timestamp",
        "reporting_path",
        "gpu_id",
    ]

    context = dict(zip(names, args))
    context.update(kwargs)

    return context


def setup_child_run_live_logging(
    experiment_name: str,
    cfg: dict,
    run_id: int,
    timestamp: str,
    reporting_path: Path,
    gpu_id: int | None,
):
    """
    Redirect the child process stdout/stderr file descriptors into its own
    run_live.log and return a cleanup function.

    This captures Python prints, logging output, many native-library messages,
    and subprocess output emitted by the child. The parent dashboard is not
    polluted by child output, and each run has a separate live-readable log file.
    """
    run_log_path = build_run_live_log_path(
        reporting_path=Path(reporting_path),
        timestamp=str(timestamp),
        experiment_name=experiment_name,
        cfg=cfg,
        run_id=int(run_id),
    )

    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Flush before changing file descriptors so buffered terminal output does
    # not get mixed into the run log.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    stdout_backup_fd = os.dup(1)
    stderr_backup_fd = os.dup(2)
    run_log_file = run_log_path.open("a", encoding="utf-8", buffering=1)

    os.dup2(run_log_file.fileno(), 1)
    os.dup2(run_log_file.fileno(), 2)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass

    print("=" * 110, flush=True)
    print(
        f"[{utc_timestamp_for_log()}] START "
        f"experiment={experiment_name} run_{run_id} "
        f"pid={os.getpid()} gpu={gpu_id} log={run_log_path}",
        flush=True,
    )
    print("=" * 110, flush=True)

    def cleanup(status: str = "finished"):
        try:
            print("=" * 110, flush=True)
            print(
                f"[{utc_timestamp_for_log()}] END "
                f"experiment={experiment_name} run_{run_id} "
                f"pid={os.getpid()} status={status}",
                flush=True,
            )
            print("=" * 110, flush=True)
        except Exception:
            pass

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        try:
            os.dup2(stdout_backup_fd, 1)
            os.dup2(stderr_backup_fd, 2)
        except Exception:
            pass

        for fd in (stdout_backup_fd, stderr_backup_fd):
            try:
                os.close(fd)
            except Exception:
                pass

        try:
            run_log_file.flush()
            run_log_file.close()
        except Exception:
            pass

    return cleanup, run_log_path


# =============================================================================
# Small helpers
# =============================================================================

def safe_experiment_name(name: str) -> str:
    return "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in str(name)
    )


def cfg_needs_gpu(cfg: dict) -> bool:
    """
    Decide whether this experiment should wait for GPU capacity.

    Rule:
        Every training_type except None / null / "" should use GPU.
    """
    training_type = cfg.get("sim", {}).get("training_type", None)

    if training_type is None:
        return False

    training_type_str = str(training_type).strip().lower()

    if training_type_str in {"", "none", "null"}:
        return False

    return True


def find_cfg_value(cfg, candidate_keys, default="NA"):
    """
    Recursively search a nested config dict/list for the first matching key.

    This is kept as a fallback only. For frequency/noise/lead in folder names,
    build_experiment_folder_name first reads the exact YAML paths used by your
    config, because names like seasonality_frequncy and random_walk.variance are
    otherwise easy to miss.
    """
    candidate_keys = {str(k).lower() for k in candidate_keys}

    def key_matches(key) -> bool:
        key_lower = str(key).lower()

        if key_lower in candidate_keys:
            return True

        return any(candidate_key in key_lower for candidate_key in candidate_keys)

    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if key_matches(key):
                return value

        for value in cfg.values():
            found = find_cfg_value(value, candidate_keys, default=None)
            if found is not None:
                return found

    elif isinstance(cfg, list):
        for item in cfg:
            found = find_cfg_value(item, candidate_keys, default=None)
            if found is not None:
                return found

    return default


def get_cfg_path(cfg: dict, path: list[str], default="NA"):
    """
    Read an exact nested path from the already merged experiment config.

    Example:
        get_cfg_path(cfg, ["market", "random_walk", "variance"])
    """
    current = cfg

    for key in path:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    if current is None:
        return default

    return current


def first_existing_cfg_path(cfg: dict, paths: list[list[str]], default="NA"):
    """
    Return the first non-empty value from several exact config paths.
    """
    for path in paths:
        value = get_cfg_path(cfg, path, default=None)

        if value is None:
            continue

        if isinstance(value, str) and value.strip() == "":
            continue

        return value

    return default


def get_supply_chain_lead_value(cfg: dict, default="NA"):
    """
    Read lead_time from supply_chain.sc_levels.

    If all levels have the same lead_time, return one value, for example 1.
    If levels differ, return a compact joined value, for example 1-2-3.
    """
    sc_levels = get_cfg_path(
        cfg,
        ["supply_chain", "sc_levels"],
        default=None,
    )

    if not isinstance(sc_levels, dict):
        return first_existing_cfg_path(
            cfg,
            paths=[
                ["supply_chain", "lead_time"],
                ["supply_chain", "leadtime"],
                ["market", "lead_time"],
            ],
            default=default,
        )

    lead_values = []

    for level_name in sorted(sc_levels.keys()):
        level_cfg = sc_levels[level_name]

        if not isinstance(level_cfg, dict):
            continue

        if "lead_time" in level_cfg:
            lead_values.append(level_cfg["lead_time"])
        elif "leadtime" in level_cfg:
            lead_values.append(level_cfg["leadtime"])

    if not lead_values:
        return default

    # If the same lead time is used on all levels, keep the folder name short.
    if all(value == lead_values[0] for value in lead_values):
        return lead_values[0]

    return "-".join(compact_value_for_path(value) for value in lead_values)

def compact_value_for_path(value) -> str:
    """
    Convert a config value into a safe short string for folder names.
    """
    if value is None:
        return "NA"

    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)

    text = text.strip()
    text = text.replace(" ", "")
    text = text.replace("/", "-")
    text = text.replace("\\", "-")
    text = text.replace(":", "-")
    text = text.replace(",", "-")

    return safe_experiment_name(text)


def get_run_start_id(cfg: dict, default: int = 0) -> int:
    """
    Read the first run ID to use for this experiment from the config.

    Preferred YAML location:
        sim.run_start_id

    Fallback location, also accepted:
        run_start_id

    Example:
        simulation_runs: 3
        run_start_id: 10

    This creates runs 10, 11, 12 instead of 0, 1, 2.
    The run_id is also used in the seed, so later batches can continue with
    new seeds by increasing run_start_id.
    """
    value = first_existing_cfg_path(
        cfg,
        paths=[
            ["sim", "run_start_id"],
            ["run_start_id"],
        ],
        default=default,
    )

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"run_start_id must be an integer-compatible value, got {value!r}"
        ) from exc


def get_experiment_dashboard_values(cfg: dict) -> dict:
    """
    Extract values that should be shown in the CLI dashboard and logs.

    For your YAML:
        frequency -> market.seasonality_frequncy
        noise     -> market.random_walk.variance
        magnitude -> first matching magnitude path/key
        lead_time -> supply_chain.sc_levels.*.lead_time
    """
    freq = first_existing_cfg_path(
        cfg,
        paths=[
            ["market", "seasonality_frequncy"],
            ["market", "seasonality_frequency"],
            ["market", "frequency"],
        ],
        default="NA",
    )

    noise = first_existing_cfg_path(
        cfg,
        paths=[
            ["market", "random_walk", "variance"],
            ["market", "random_walk", "std"],
            ["market", "random_walk", "sigma"],
            ["market", "noise_level"],
            ["market", "noise"],
        ],
        default="NA",
    )

    magnitude = first_existing_cfg_path(
        cfg,
        paths=[
            ["market", "demand_magnitude"],
            ["market", "seasonality_magnitude"],
            ["market", "magnitude"],
            ["market", "base_demand_magnitude"],
            ["demand_magnitude"],
            ["seasonality_magnitude"],
            ["magnitude"],
        ],
        default=None,
    )

    if magnitude is None:
        magnitude = find_cfg_value(
            cfg,
            candidate_keys=[
                "demand_magnitude",
                "seasonality_magnitude",
                "base_demand_magnitude",
                "magnitude",
            ],
            default="NA",
        )

    lead_time = get_supply_chain_lead_value(cfg, default="NA")

    dataset_name = first_existing_cfg_path(
        cfg,
        paths=[
            ["market", "dataset_name"],
            ["market", "product_name"],
        ],
        default="NA",
    )

    data_source = first_existing_cfg_path(
        cfg,
        paths=[
            ["market", "data_scource"],
            ["market", "data_source"],
        ],
        default="NA",
    )

    return {
        "freq": compact_value_for_path(freq),
        "noise": compact_value_for_path(noise),
        "magnitude": compact_value_for_path(magnitude),
        "lead_time": compact_value_for_path(lead_time),
        "dataset": compact_value_for_path(dataset_name),
        "data_source": compact_value_for_path(data_source),
    }

def build_experiment_folder_name(experiment_name: str, cfg: dict) -> str:
    """
    Final experiment folder format:
        <experiment_name>

    Example:
        timesfm_zero_shot_001

    The frequency/noise/magnitude/lead_time values are shown in the CLI dashboard,
    not duplicated in the folder name.
    """
    return safe_experiment_name(experiment_name)


# =============================================================================
# Model checkpointing
# =============================================================================

# Only these training types produce newly trained model weights that are useful
# to save for later analysis. Zero-shot/pretrained and no-training baselines are
# intentionally excluded.
MODEL_SAVE_TRAINING_TYPES = {
    "local_multichannel",
    "split_multichannel",
    "local_timemixer",
    "split_timemixer",
    "split_timemixer_option_d",  # accepted as an alias/safety fallback
    "local_patchtst",
    "split_patchtst",
}
MODEL_SAVE_TRAINING_TYPES = {}


def normalize_training_type(cfg: dict) -> str:
    """
    Read cfg['sim']['training_type'] as a normalized lowercase string.
    """
    training_type = cfg.get("sim", {}).get("training_type", None)

    if training_type is None:
        return ""

    return str(training_type).strip().lower()


def should_save_model_for_cfg(cfg: dict) -> bool:
    """
    Decide whether this experiment should write a trained-model checkpoint.
    """
    return normalize_training_type(cfg) in MODEL_SAVE_TRAINING_TYPES


def model_state_dict_cpu(model) -> dict:
    """
    Return a CPU-only copy of a torch module state_dict.

    Saving CPU tensors makes the checkpoint easier to inspect/reload on machines
    without the original CUDA device.
    """
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def module_list_state_dict_cpu(models) -> list[dict]:
    """
    Save an nn.ModuleList or normal Python list/tuple of torch modules.
    """
    return [
        model_state_dict_cpu(model)
        for model in list(models)
    ]


def get_agent_forecasting_model(agent):
    """
    Safely read the forecasting model attached to an agent.
    """
    if hasattr(agent, "get_forecasting_model"):
        try:
            return agent.get_forecasting_model()
        except Exception:
            return None

    return getattr(agent, "forecasting_model", None)


def collect_server_model_once(sc_agent_list):
    """
    For split PatchTST/TimeMixer, the same shared server_model is attached to
    the forecasting model of each participating agent. Save it once.

    split_multichannel currently has no trainable server module; in that case
    this returns None.
    """
    seen_ids = set()

    for level_agents in sc_agent_list:
        for agent in level_agents:
            fm = get_agent_forecasting_model(agent)

            if fm is None or not hasattr(fm, "server_model"):
                continue

            server_model = fm.server_model

            if server_model is None:
                continue

            if id(server_model) in seen_ids:
                continue

            seen_ids.add(id(server_model))
            return server_model

    return None


def build_agent_model_checkpoint_entry(level_idx: int, agent_idx: int, agent) -> dict | None:
    """
    Export the trainable parts of one agent's attached forecasting model.

    This supports the current model containers used by:
      - local_multichannel / split_multichannel: lstm_model + dense_model
      - local_timemixer / local_patchtst: models
      - split_timemixer / split_patchtst: client_models + shared server_model
    """
    fm = get_agent_forecasting_model(agent)

    if fm is None:
        return None

    entry = {
        "level_idx": level_idx,
        "agent_idx": agent_idx,
        "num_retailer": getattr(agent, "num_retailer", None),
        "sequence_length": getattr(agent, "sequence_length", None),
        "forecasting_model_class": type(fm).__name__,
    }

    saved_any_model = False

    # local_multichannel / split_multichannel
    if hasattr(fm, "lstm_model") and fm.lstm_model is not None:
        entry["lstm_model"] = module_list_state_dict_cpu(fm.lstm_model)
        saved_any_model = True

    if hasattr(fm, "dense_model") and fm.dense_model is not None:
        entry["dense_model"] = module_list_state_dict_cpu(fm.dense_model)
        saved_any_model = True

    # local_timemixer / local_patchtst
    if hasattr(fm, "models") and fm.models is not None:
        entry["models"] = module_list_state_dict_cpu(fm.models)
        saved_any_model = True

    # split_timemixer / split_patchtst client-side modules
    if hasattr(fm, "client_models") and fm.client_models is not None:
        entry["client_models"] = module_list_state_dict_cpu(fm.client_models)
        saved_any_model = True

    # The scaler is required to interpret model outputs later.
    # torch.save can pickle sklearn StandardScaler objects.
    if hasattr(fm, "scaler"):
        entry["scaler"] = fm.scaler

    # Keep lightweight architecture metadata that may help with later analysis.
    for attr in [
        "horizon",
        "scales",
        "scale_lengths",
        "num_patches",
        "device",
    ]:
        if hasattr(fm, attr):
            value = getattr(fm, attr)
            entry[attr] = str(value) if attr == "device" else value

    if not saved_any_model:
        return None

    return entry


def save_trained_forecasting_models(
    cfg: dict,
    sc_agent_list,
    output_path: Path,
    run_id: int,
    val_loss,
):
    """
    Save trained forecasting models for the selected local/split training types.

    Final files:
        <run_folder>/model/trained_models.pt
        <run_folder>/model/trained_models_metadata.json

    The .pt file is the analysis checkpoint. It includes:
      - cfg
      - run_id
      - val_loss
      - per-agent trainable modules
      - per-agent scalers
      - shared server_model for split PatchTST/TimeMixer, saved once
    """
    training_type = normalize_training_type(cfg)

    if training_type not in MODEL_SAVE_TRAINING_TYPES:
        return

    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": 1,
        "training_type": training_type,
        "run_id": run_id,
        "val_loss": val_loss,
        "cfg": cfg,
        "agents": [],
    }

    for level_idx, level_agents in enumerate(sc_agent_list):
        for agent_idx, agent in enumerate(level_agents):
            entry = build_agent_model_checkpoint_entry(
                level_idx=level_idx,
                agent_idx=agent_idx,
                agent=agent,
            )

            if entry is not None:
                checkpoint["agents"].append(entry)

    server_model = collect_server_model_once(sc_agent_list)

    if server_model is not None:
        checkpoint["server_model_class"] = type(server_model).__name__
        checkpoint["server_model"] = model_state_dict_cpu(server_model)

    checkpoint_path = output_path / "trained_models.pt"
    torch.save(checkpoint, checkpoint_path)

    metadata = {
        "format_version": 1,
        "training_type": training_type,
        "run_id": run_id,
        "saved_file": checkpoint_path.name,
        "contains_server_model": server_model is not None,
        "server_model_class": type(server_model).__name__ if server_model is not None else None,
        "num_saved_agents": len(checkpoint["agents"]),
        "agent_model_classes": sorted(
            {
                entry.get("forecasting_model_class", "unknown")
                for entry in checkpoint["agents"]
            }
        ),
    }

    metadata_path = output_path / "trained_models_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def save_config_once(experiment_reporting_path: Path, cfg: dict):
    """
    Save the specific config for this experiment once in the experiment folder.

    Final location:
        Reporting/<parent_timestamp>/<experiment_folder>/config.json

    Multiple run processes may try to write this at the same time, so this uses
    a temp file and atomic replace.
    """
    experiment_reporting_path.mkdir(parents=True, exist_ok=True)

    config_json_path = experiment_reporting_path / "config.json"

    if config_json_path.exists():
        return

    tmp_path = experiment_reporting_path / f".config_{os.getpid()}.tmp"

    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(
                cfg,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        os.replace(tmp_path, config_json_path)

    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def is_artifact_file_name(name: str) -> bool:
    """
    Files that should not remain inside individual run folders.
    """
    normalized_name = name.lower()

    return normalized_name in {
        "config.json",
        "config.yaml",
        "reporting_overview.csv",
        "bew_measures.csv",
        "bew_measures.json",
    }


def merge_directory_contents(source_dir: Path, target_dir: Path):
    """
    Move all non-global artifacts from source_dir into target_dir.

    This is used to flatten whatever nested layout Reporting creates into:
        experiment_folder/run_0/<images/data/...>
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists() or not source_dir.is_dir():
        return

    if source_dir.resolve() == target_dir.resolve():
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    for item in list(source_dir.iterdir()):
        if is_artifact_file_name(item.name):
            continue

        # Skip internal wrapper folders; their contents are handled by selecting
        # the deepest useful run directory below.
        if item.is_dir() and item.name.lower().startswith("simulation_run"):
            continue

        destination = target_dir / item.name

        if destination.exists():
            if item.is_dir() and destination.is_dir():
                merge_directory_contents(item, destination)
                try:
                    item.rmdir()
                except OSError:
                    pass
            elif item.is_file() and destination.is_file():
                # Keep the first file and avoid overwriting data from another run.
                item.unlink()
            else:
                fallback_destination = target_dir / (
                    f"{item.stem}_{os.getpid()}{item.suffix}"
                    if item.is_file()
                    else f"{item.name}_{os.getpid()}"
                )
                shutil.move(str(item), str(fallback_destination))
        else:
            shutil.move(str(item), str(destination))


def append_unique_csv(source_csv_path: Path, target_csv_path: Path):
    """
    Append source_csv_path into target_csv_path while keeping one header.

    The scheduler runs experiments in parallel, so this uses a simple lock file
    around the append operation on Linux. If fcntl is unavailable, it still works
    best-effort without the lock.
    """
    source_csv_path = Path(source_csv_path)
    target_csv_path = Path(target_csv_path)

    if not source_csv_path.exists() or not source_csv_path.is_file():
        return

    source_lines = source_csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    if not source_lines:
        return

    target_csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_csv_path.with_suffix(target_csv_path.suffix + ".lock")

    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except Exception:
            pass

        if target_csv_path.exists():
            target_lines = target_csv_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
        else:
            target_lines = []

        if not target_lines:
            lines_to_write = source_lines
        else:
            target_header = target_lines[0]
            source_header = source_lines[0]

            if source_header == target_header:
                data_lines = source_lines[1:]
            else:
                # Different header: keep the whole source block rather than
                # silently dropping information.
                data_lines = source_lines

            existing_data_lines = set(target_lines[1:])
            lines_to_write = [
                line for line in data_lines
                if line and line not in existing_data_lines
            ]

        if lines_to_write:
            with target_csv_path.open("a", encoding="utf-8", newline="") as f:
                if target_lines:
                    f.write("\n")
                f.write("\n".join(lines_to_write))

        try:
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        except Exception:
            pass


def move_reporting_overview_to_parent(
    raw_reporting_path: Path,
    parent_reporting_path: Path,
):
    """
    Put all generated reporting_overview.csv rows into the timestamp parent.

    Final location:
        Reporting/<parent_timestamp>/reporting_overview.csv
    """
    target = parent_reporting_path / "reporting_overview.csv"

    for source in raw_reporting_path.rglob("reporting_overview.csv"):
        if source.resolve() == target.resolve():
            continue

        append_unique_csv(
            source_csv_path=source,
            target_csv_path=target,
        )


def copy_bew_measures_to_experiment_folder(
    raw_reporting_path: Path,
    experiment_reporting_path: Path,
):
    """
    Copy one BEW_measures.csv into the experiment folder.

    Final location:
        Reporting/<parent_timestamp>/<experiment_folder>/BEW_measures.csv
    """
    target = experiment_reporting_path / "BEW_measures.csv"

    if target.exists():
        return

    candidate_names = {
        "bew_measures.csv",
        "bew_measures.json",
    }

    for source in raw_reporting_path.rglob("*"):
        if not source.is_file():
            continue

        if source.name.lower() not in candidate_names:
            continue

        tmp_target = experiment_reporting_path / f".BEW_measures_{os.getpid()}.tmp"
        try:
            shutil.copy2(source, tmp_target)
            os.replace(tmp_target, target)
        finally:
            if tmp_target.exists():
                try:
                    tmp_target.unlink()
                except Exception:
                    pass
        return


def directory_contains_run_payload(path: Path) -> bool:
    """
    Detect a directory that contains the actual per-run output payload.
    """
    if not path.exists() or not path.is_dir():
        return False

    child_names = {child.name.lower() for child in path.iterdir()}

    if "images" in child_names or "data" in child_names:
        return True

    return any(
        child.is_file()
        and child.name.lower() not in {
            "config.json",
            "config.yaml",
            "reporting_overview.csv",
            "bew_measures.csv",
        }
        for child in path.iterdir()
    )


def find_best_run_payload_dir(
    raw_reporting_path: Path,
    run_folder_name: str,
) -> Path:
    """
    Find the directory created by Reporting that contains the actual files for
    this run, even if Reporting nested it under simulation_runs_*/run_*.
    """
    candidates = [raw_reporting_path]

    candidates.extend(
        path for path in raw_reporting_path.rglob("*")
        if path.is_dir()
        and (
            path.name == run_folder_name
            or path.name.lower().startswith("simulation_run")
        )
    )

    def score(path: Path):
        depth = len(path.relative_to(raw_reporting_path).parts) if path != raw_reporting_path else 0
        name_match = 1 if path.name == run_folder_name else 0
        has_payload = 1 if directory_contains_run_payload(path) else 0

        return (has_payload, name_match, depth)

    candidates.sort(key=score, reverse=True)

    return candidates[0]


def normalize_reporting_output(
    raw_reporting_path: Path,
    parent_reporting_path: Path,
    experiment_reporting_path: Path,
    run_folder_name: str,
):
    """
    Normalize Reporting's generated output into the requested final layout:

    Reporting/
      <parent_timestamp>/
        reporting_overview.csv
        <experiment_folder>/
          config.json
          BEW_measures.csv
          run_0/
            images/
            data/
          run_1/
            images/
            data/
    """
    parent_reporting_path.mkdir(parents=True, exist_ok=True)
    experiment_reporting_path.mkdir(parents=True, exist_ok=True)

    desired_run_path = experiment_reporting_path / run_folder_name
    desired_run_path.mkdir(parents=True, exist_ok=True)

    move_reporting_overview_to_parent(
        raw_reporting_path=raw_reporting_path,
        parent_reporting_path=parent_reporting_path,
    )

    copy_bew_measures_to_experiment_folder(
        raw_reporting_path=raw_reporting_path,
        experiment_reporting_path=experiment_reporting_path,
    )

    best_payload_dir = find_best_run_payload_dir(
        raw_reporting_path=raw_reporting_path,
        run_folder_name=run_folder_name,
    )

    merge_directory_contents(
        source_dir=best_payload_dir,
        target_dir=desired_run_path,
    )


# =============================================================================
# GPU monitoring and allocation
# =============================================================================

def parse_gpu_tokens(value: str | None) -> list[str]:
    """
    Parse GPU lists such as:
        0,1,2
        2,4,7
        0-2
        GPU-abc,GPU-def
        MIG-GPU-...

    Numeric ranges are expanded. UUID/MIG tokens remain unchanged.
    """
    if value is None:
        return []

    value = str(value).strip()

    if not value or value.lower() in {"none", "nodevfiles", "(null)"}:
        return []

    tokens: list[str] = []

    for part in value.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part and not part.startswith(("GPU-", "MIG-")):
            bounds = part.split("-", maxsplit=1)

            if len(bounds) == 2 and all(bound.isdigit() for bound in bounds):
                start, end = map(int, bounds)

                if end >= start:
                    tokens.extend(str(index) for index in range(start, end + 1))
                    continue

        tokens.append(part)

    return tokens


def query_nvidia_smi_rows() -> list[dict]:
    """
    Return physical GPU information from nvidia-smi.

    Both index and UUID are queried so fixed and Slurm allocations can be
    matched regardless of whether they use numeric IDs or UUIDs.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception:
        return []

    rows: list[dict] = []

    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue

        parts = [part.strip() for part in line.split(",")]

        if len(parts) != 5:
            continue

        index, uuid, utilization, memory_used, memory_total = parts

        try:
            memory_used_float = float(memory_used)
            memory_total_float = float(memory_total)

            rows.append(
                {
                    "index": index,
                    "uuid": uuid,
                    "gpu_usage": float(utilization),
                    "memory_used_mb": memory_used_float,
                    "memory_total_mb": memory_total_float,
                    "memory_usage": (
                        memory_used_float / memory_total_float
                        if memory_total_float > 0
                        else 1.0
                    ),
                }
            )
        except ValueError:
            continue

    return rows


def is_running_under_slurm() -> bool:
    """Return True when the current process appears to be inside a Slurm job."""
    return any(
        os.environ.get(name)
        for name in (
            "SLURM_JOB_ID",
            "SLURM_STEP_ID",
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
        )
    )


def configure_gpu_allocation(
    mode: str,
    fixed_gpu_ids: str | list[str] | None = None,
) -> list[dict]:
    """
    Configure the GPUs available to the parent scheduler.

    Modes:
        fixed:
            Use the explicit IDs supplied through --fixed-gpu-ids. This keeps
            the previous local behavior, e.g. physical GPUs 0,1,2.

        slurm:
            Use only GPUs assigned by Slurm. CUDA_VISIBLE_DEVICES determines
            the token passed to child processes; SLURM_STEP_GPUS or
            SLURM_JOB_GPUS is used to map nvidia-smi monitoring to physical IDs.

        auto:
            Use Slurm allocation when running inside Slurm, otherwise use the
            fixed IDs.

    The returned gpu_id is always a logical scheduler slot 0..N-1. A slot maps
    to a cuda_token that is used only when spawning the corresponding child.
    """
    global GPU_ALLOCATION_MODE, FIXED_GPU_IDS, GPU_SLOTS

    normalized_mode = str(mode).strip().lower()

    if normalized_mode not in {"fixed", "slurm", "auto"}:
        raise ValueError(
            "gpu allocation mode must be one of: fixed, slurm, auto"
        )

    if isinstance(fixed_gpu_ids, str):
        parsed_fixed_ids = parse_gpu_tokens(fixed_gpu_ids)
    elif fixed_gpu_ids is None:
        parsed_fixed_ids = list(FIXED_GPU_IDS)
    else:
        parsed_fixed_ids = [str(value) for value in fixed_gpu_ids]

    if normalized_mode == "auto":
        resolved_mode = "slurm" if is_running_under_slurm() else "fixed"
    else:
        resolved_mode = normalized_mode

    GPU_ALLOCATION_MODE = resolved_mode
    FIXED_GPU_IDS = parsed_fixed_ids

    if resolved_mode == "fixed":
        if not parsed_fixed_ids:
            raise RuntimeError(
                "Fixed GPU allocation selected, but --fixed-gpu-ids is empty."
            )

        cuda_tokens = list(parsed_fixed_ids)
        monitor_tokens = list(parsed_fixed_ids)

    else:
        cuda_tokens = parse_gpu_tokens(os.environ.get("CUDA_VISIBLE_DEVICES"))
        slurm_tokens = parse_gpu_tokens(
            os.environ.get("SLURM_STEP_GPUS")
            or os.environ.get("SLURM_JOB_GPUS")
        )

        if not cuda_tokens and slurm_tokens:
            cuda_tokens = list(slurm_tokens)

        if not cuda_tokens:
            raise RuntimeError(
                "Slurm GPU allocation selected, but neither "
                "CUDA_VISIBLE_DEVICES nor SLURM_STEP_GPUS/SLURM_JOB_GPUS "
                "contains an allocation. Request GPUs with --gres/--gpus."
            )

        # SLURM_*_GPUS usually contains global/physical IDs and is therefore
        # preferable for matching nvidia-smi. CUDA_VISIBLE_DEVICES is retained
        # as the token inherited by each child process.
        if slurm_tokens and len(slurm_tokens) == len(cuda_tokens):
            monitor_tokens = slurm_tokens
        else:
            monitor_tokens = list(cuda_tokens)

    GPU_SLOTS = [
        {
            "gpu_id": logical_id,
            "cuda_token": str(cuda_token),
            "monitor_token": str(monitor_token),
        }
        for logical_id, (cuda_token, monitor_token) in enumerate(
            zip(cuda_tokens, monitor_tokens)
        )
    ]

    return GPU_SLOTS


def get_gpu_stats() -> list[dict]:
    """
    Return statistics only for GPUs configured for this scheduler.

    gpu_id is always a logical slot. In fixed mode it maps to the requested
    fixed ID; in Slurm mode it maps to one GPU assigned to the job.
    """
    if not GPU_SLOTS:
        return []

    rows = query_nvidia_smi_rows()

    if not rows:
        return []

    rows_by_token: dict[str, dict] = {}

    for row in rows:
        rows_by_token[str(row["index"])] = row
        rows_by_token[str(row["uuid"])] = row

    stats: list[dict] = []

    for slot in GPU_SLOTS:
        row = rows_by_token.get(str(slot["monitor_token"]))

        # Some Slurm/cgroup setups make nvidia-smi show only the allocated
        # devices. When counts match, row order is the safest fallback.
        if row is None and len(rows) == len(GPU_SLOTS):
            row = rows[slot["gpu_id"]]

        if row is None:
            # Do not schedule onto an allocation that cannot be monitored and
            # safely mapped.
            continue

        stats.append(
            {
                "gpu_id": slot["gpu_id"],
                "cuda_token": slot["cuda_token"],
                "monitor_token": slot["monitor_token"],
                "gpu_usage": row["gpu_usage"],
                "memory_used_mb": row["memory_used_mb"],
                "memory_total_mb": row["memory_total_mb"],
                "memory_usage": row["memory_usage"],
            }
        )

    return stats


def get_gpu_usage():
    """Compatibility wrapper returning utilization by logical GPU slot."""
    return [gpu["gpu_usage"] for gpu in get_gpu_stats()]


def get_gpu_memory_usage():
    """Compatibility wrapper returning memory by logical GPU slot."""
    return [
        {
            "used_mb": gpu["memory_used_mb"],
            "total_mb": gpu["memory_total_mb"],
            "memory_usage": gpu["memory_usage"],
        }
        for gpu in get_gpu_stats()
    ]


def choose_gpu_for_new_process(
    max_gpu_usage: float = 70.0,
    max_gpu_memory_usage: float = 0.85,
    active_gpu_jobs: dict | None = None,
    max_jobs_per_gpu: int = 1,
):
    """
    Choose one logical GPU slot from the configured fixed/Slurm allocation.
    """
    if active_gpu_jobs is None:
        active_gpu_jobs = {}

    candidates = []

    for gpu in get_gpu_stats():
        gpu_id = gpu["gpu_id"]
        running_jobs = active_gpu_jobs.get(gpu_id, 0)

        if running_jobs >= max_jobs_per_gpu:
            continue

        if gpu["gpu_usage"] > max_gpu_usage:
            continue

        if gpu["memory_usage"] > max_gpu_memory_usage:
            continue

        candidates.append(
            {
                "gpu_id": gpu_id,
                "gpu_usage": gpu["gpu_usage"],
                "memory_usage": gpu["memory_usage"],
                "running_jobs": running_jobs,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["running_jobs"],
            item["memory_usage"],
            item["gpu_usage"],
        )
    )

    return candidates[0]["gpu_id"]


# =============================================================================
# Live CLI dashboard
# =============================================================================

def clear_terminal():
    """
    Clear terminal and move cursor to top-left.

    Works in normal Linux terminals/SSH.
    Some IDE consoles may still print repeated blocks.
    """
    print("\033[2J\033[H", end="", flush=True)


def render_dashboard(
    total_jobs: int,
    done_jobs: int,
    failed_jobs_count: int,
    pending_jobs,
    active_processes: dict,
    active_gpu_jobs: dict,
    last_event: str = "",
):
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = psutil.virtual_memory().percent
    gpu_stats = get_gpu_stats()

    clear_terminal()

    print("=" * 110)
    print("EXPERIMENT SCHEDULER")
    print("=" * 110)

    print(
        f"Jobs: done={done_jobs}/{total_jobs} | "
        f"running={len(active_processes)} | "
        f"waiting={len(pending_jobs)} | "
        f"failed={failed_jobs_count}"
    )

    print(f"CPU: {cpu_usage:5.1f}% | RAM: {ram_usage:5.1f}%")
    print("-" * 110)

    if gpu_stats:
        for gpu in gpu_stats:
            gpu_id = gpu["gpu_id"]
            print(
                f"GPU {gpu_id}: "
                f"util={gpu['gpu_usage']:5.1f}% | "
                f"mem={gpu['memory_used_mb']:7.0f}/{gpu['memory_total_mb']:.0f} MB "
                f"({gpu['memory_usage'] * 100:5.1f}%) | "
                f"jobs={active_gpu_jobs.get(gpu_id, 0)}"
            )
    else:
        print("GPU: no stats available")

    print("-" * 110)
    print(f"Last: {last_event if last_event else '-'}")
    print("-" * 110)

    if active_processes:
        print("Currently running:")
        for pid, info in active_processes.items():
            job = info["job"]
            run_id_label = job.get("run_id_label", f"run_{job['run_id']}")
            print(
                f"  PID {pid} | "
                f"{job.get('training_type')} | "
                f"{job['experiment_name']} | "
                f"{run_id_label} | "
                f"freq={job.get('freq', 'NA')} | "
                f"noise={job.get('noise', 'NA')} | "
                f"magnitude={job.get('magnitude', 'NA')} | "
                f"lead_time={job.get('lead_time', 'NA')} | "
                f"GPU {info['gpu_id']}"
            )
    else:
        print("Currently running: none")

    print("=" * 110)
    print("Press Ctrl+C to stop.")
    sys.stdout.flush()


def configure_jax_for_current_worker(cfg: dict, gpu_assigned: bool):
    """
    Configure JAX before ML_Backends/TimesFM is imported.

    Explicit YAML values win:
        timesfm.jax_platforms: cpu
        timesfm.jax_platforms: cuda

    If the value is missing or set to auto, Flax TimesFM uses CUDA when this
    worker has a GPU and CPU otherwise.
    """
    sim_cfg = cfg.get("sim", {})
    timesfm_cfg = cfg.get("timesfm", {})

    training_type = str(sim_cfg.get("training_type", "")).strip().lower()
    backend = str(timesfm_cfg.get("backend", "")).strip().lower()

    if training_type != "timesfm_zero_shot" or backend != "flax":
        return

    configured_platform = str(
        timesfm_cfg.get("jax_platforms", "auto")
    ).strip().lower()

    if configured_platform in {"", "auto", "default"}:
        configured_platform = "cuda" if gpu_assigned else "cpu"

    os.environ["JAX_PLATFORMS"] = configured_platform
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


# =============================================================================
# Device selection inside child workers
# =============================================================================

def select_device_for_current_worker(gpu_assigned: bool):
    """
    Used to patch helpers.helpers.select_gpu inside the child process.

    Because CUDA_VISIBLE_DEVICES is set before the child starts,
    the assigned physical GPU appears as cuda:0 inside the child.
    """
    if gpu_assigned and torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def patch_select_gpu_for_worker(gpu_id):
    """
    Patch helpers.helpers.select_gpu before importing ML_Backends.
    """
    import helpers.helpers as helper_module

    def scheduler_selected_gpu():
        return select_device_for_current_worker(gpu_assigned=gpu_id is not None)

    helper_module.select_gpu = scheduler_selected_gpu


# =============================================================================
# Single experiment worker
# =============================================================================

def run_single_experiment(
    experiment_id: int,
    total_experiments: int,
    experiment_name: str,
    cfg: dict,
    run_id: int,
    timestamp: str,
    reporting_path: Path,
    gpu_id: int | None = None,
):
    """
    Runs exactly one experiment/run pair inside a child process.
    """

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    configure_jax_for_current_worker(cfg, gpu_assigned=gpu_id is not None)

    patch_select_gpu_for_worker(gpu_id)

    # Lazy imports to avoid circular import:
    # helpers.helpers -> ML_Backends -> lstm_local -> helpers.helpers
    from Simulation_Component.reporting import Reporting
    from Simulation_Component.runner import run_simulation_phase
    from ML_Backends import build_backend
    from ML_Backends.ma import NoOpBackend

    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(0)

    seed = 42 + experiment_id * 100_000 + run_id

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    simulation, market, supply_chain, sc_agent_list = reset_simulaltion_from_dict(cfg)

    backend = build_backend(cfg)

    test_end = simulation.conv_time + simulation.sim_time + simulation.testing_time

    warmup_end = (
        simulation.conv_time + simulation.sim_time
        if backend.needs_training_phase
        else test_end
    )

    run_simulation_phase(
        0,
        warmup_end,
        simulation,
        market,
        supply_chain,
        sc_agent_list,
        NoOpBackend(cfg),
    )

    val_loss = backend.train(
        simulation,
        market,
        supply_chain,
        sc_agent_list,
    )

    if warmup_end < test_end:
        run_simulation_phase(
            warmup_end,
            test_end,
            simulation,
            market,
            supply_chain,
            sc_agent_list,
            backend,
        )

    # -------------------------------------------------------------------------
    # Requested reporting folder structure:
    #
    # Reporting/
    #   <parent timestamp>/
    #     reporting_overview.csv
    #     <experiment_name>/
    #       config.json
    #       BEW_measures.csv
    #       run_0/
    #         images/
    #         data/
    #       run_1/
    #         images/
    #         data/
    # -------------------------------------------------------------------------

    parent_reporting_path = reporting_path / timestamp

    experiment_folder_name = build_experiment_folder_name(
        experiment_name=experiment_name,
        cfg=cfg,
    )

    experiment_reporting_path = parent_reporting_path / experiment_folder_name

    parent_reporting_path.mkdir(parents=True, exist_ok=True)
    experiment_reporting_path.mkdir(parents=True, exist_ok=True)

    save_config_once(
        experiment_reporting_path=experiment_reporting_path,
        cfg=cfg,
    )

    run_folder_name = f"run_{run_id}"
    run_reporting_path = experiment_reporting_path / run_folder_name

    save_trained_forecasting_models(
        cfg=cfg,
        sc_agent_list=sc_agent_list,
        output_path=run_reporting_path / "model",
        run_id=run_id,
        val_loss=val_loss,
    )

    # Let the existing Reporting class write wherever it normally writes, but
    # isolate that output in a hidden raw folder first. Then normalize it into
    # the clean structure above. This prevents extra simulation_runs_* folders,
    # duplicate run_* folders, duplicate configs, and per-run overview files from
    # leaking into the final report folder.
    raw_reporting_path = (
        experiment_reporting_path
        / f".raw_{run_folder_name}_{os.getpid()}"
    )

    if raw_reporting_path.exists():
        shutil.rmtree(raw_reporting_path, ignore_errors=True)

    raw_reporting_path.mkdir(parents=True, exist_ok=True)

    try:
        Reporting(
            path=raw_reporting_path,
            timestamp=run_folder_name,
        ).create_reporting_multiple_runs(
            agent_list=sc_agent_list,
            market=market,
            supply_chain=supply_chain,
            cfg=cfg,
            run_id=run_id,
            val_loss=val_loss,
        )

        normalize_reporting_output(
            raw_reporting_path=raw_reporting_path,
            parent_reporting_path=parent_reporting_path,
            experiment_reporting_path=experiment_reporting_path,
            run_folder_name=run_folder_name,
        )

    finally:
        shutil.rmtree(raw_reporting_path, ignore_errors=True)


def run_single_experiment_process_entry(*args, **kwargs):
    """
    Hard process wrapper.

    Important:
        Do not call torch.cuda.synchronize(), torch.cuda.empty_cache(),
        gc.collect(), or logging.shutdown() here.

    Those cleanup calls can hang with CUDA/model libraries.

    os._exit(...) immediately terminates the child process after the experiment
    finishes or fails. The OS/NVIDIA driver then releases the CUDA context and
    GPU memory.

    The wrapper also sets up per-run live logging before the experiment starts,
    so any traceback produced here is written to:
        Reporting/<timestamp>/<experiment_folder>/run_<id>/run_live.log
    """
    cleanup_live_logging = None
    exit_code = 0
    status = "finished"

    try:
        context = get_process_entry_context(args, kwargs)

        cleanup_live_logging, _run_log_path = setup_child_run_live_logging(
            experiment_name=context["experiment_name"],
            cfg=context["cfg"],
            run_id=context["run_id"],
            timestamp=context["timestamp"],
            reporting_path=context["reporting_path"],
            gpu_id=context.get("gpu_id"),
        )

        run_single_experiment(*args, **kwargs)

    except BaseException:
        exit_code = 1
        status = "failed"

        try:
            logger.exception("Experiment process failed.")
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

    finally:
        if cleanup_live_logging is not None:
            try:
                cleanup_live_logging(status=status)
            except Exception:
                pass
        else:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass

        os._exit(exit_code)

# =============================================================================
# Process management
# =============================================================================

def start_process_with_gpu_env(process, gpu_id):
    """
    Start a child with exactly one configured GPU.

    gpu_id is a logical scheduler slot. In fixed mode it maps to the configured
    fixed ID; in Slurm mode it maps to a CUDA token assigned by Slurm. Inside
    the child the selected device is always visible as cuda:0.
    """
    old_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    old_cuda_device_order = os.environ.get("CUDA_DEVICE_ORDER")

    try:
        if gpu_id is not None:
            if gpu_id < 0 or gpu_id >= len(GPU_SLOTS):
                raise RuntimeError(
                    f"Invalid logical GPU slot {gpu_id}; "
                    f"available slots are 0..{len(GPU_SLOTS) - 1}."
                )

            cuda_token = GPU_SLOTS[gpu_id]["cuda_token"]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_token)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        process.start()

    finally:
        if old_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible_devices

        if old_cuda_device_order is None:
            os.environ.pop("CUDA_DEVICE_ORDER", None)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = old_cuda_device_order

def collect_finished_processes(
    active_processes: dict,
    active_gpu_jobs: dict,
):
    failed_jobs = []
    finished_jobs = []
    last_event = ""

    for pid, info in list(active_processes.items()):
        process = info["process"]

        if process.is_alive():
            continue

        process.join(timeout=2)

        gpu_id = info["gpu_id"]
        job = info["job"]

        if gpu_id is not None:
            active_gpu_jobs[gpu_id] = max(
                0,
                active_gpu_jobs.get(gpu_id, 0) - 1,
            )

            if active_gpu_jobs[gpu_id] == 0:
                del active_gpu_jobs[gpu_id]

        run_id_label = job.get("run_id_label", f"run_{job['run_id']}")

        if process.exitcode != 0:
            failed_jobs.append(info)
            append_failure_record(
                job=job,
                pid=pid,
                exitcode=process.exitcode,
                gpu_id=gpu_id,
            )
            last_event = (
                f"FAILED {job.get('training_type')} | "
                f"{job['experiment_name']} | "
                f"{run_id_label} | "
                f"freq={job.get('freq', 'NA')} | "
                f"noise={job.get('noise', 'NA')} | "
                f"magnitude={job.get('magnitude', 'NA')} | "
                f"lead_time={job.get('lead_time', 'NA')} | "
                f"exitcode={process.exitcode}"
            )
        else:
            finished_jobs.append(info)
            last_event = (
                f"Finished {job.get('training_type')} | "
                f"{job['experiment_name']} | "
                f"{run_id_label} | "
                f"freq={job.get('freq', 'NA')} | "
                f"noise={job.get('noise', 'NA')} | "
                f"magnitude={job.get('magnitude', 'NA')} | "
                f"lead_time={job.get('lead_time', 'NA')}"
            )

        del active_processes[pid]

    return failed_jobs, finished_jobs, last_event


def terminate_all_active_processes(active_processes: dict):
    """
    Clean shutdown after Ctrl+C.
    """
    if not active_processes:
        return

    print()
    print(f"Terminating {len(active_processes)} active experiment process(es)...")

    for pid, info in list(active_processes.items()):
        process = info["process"]

        if process.is_alive():
            job = info["job"]
            run_id_label = job.get("run_id_label", f"run_{job['run_id']}")
            print(
                f"Terminating PID {pid}: "
                f"{job['experiment_name']} "
                f"{run_id_label}, "
                f"freq={job.get('freq', 'NA')}, "
                f"noise={job.get('noise', 'NA')}, "
                f"magnitude={job.get('magnitude', 'NA')}, "
                f"lead_time={job.get('lead_time', 'NA')}, "
                f"GPU={info['gpu_id']}"
            )
            process.terminate()

    for pid, info in list(active_processes.items()):
        process = info["process"]
        process.join(timeout=5)

        if process.is_alive():
            print(f"Force-killing PID {pid}")
            process.kill()
            process.join(timeout=5)

    print("All known active experiment processes have been stopped.")


# =============================================================================
# Scheduler
# =============================================================================

def run_load_balanced(
    jobs: list[dict],
    max_parallel_processes: int = 4,
    max_cpu_usage: float = 75.0,
    max_ram_usage: float = 85.0,
    max_gpu_usage: float = 70.0,
    max_gpu_memory_usage: float = 0.85,
    max_jobs_per_gpu: int = 1,
    poll_seconds: float = 5.0,
):
    """
    Dynamic process scheduler.

    Starts experiment jobs only when:
    - active processes < max_parallel_processes
    - CPU usage < max_cpu_usage
    - RAM usage < max_ram_usage
    - for GPU jobs, an eligible GPU is available
    """

    gpu_jobs_exist = any(job.get("needs_gpu", False) for job in jobs)

    if gpu_jobs_exist:
        gpu_stats = get_gpu_stats()

        if not gpu_stats:
            raise RuntimeError(
                "At least one job needs a GPU, but no GPU stats are available. "
                "Check that nvidia-smi works and that CUDA/NVIDIA drivers are visible."
            )

    ctx = mp.get_context("spawn")

    pending_jobs = deque(jobs)
    active_processes = {}
    active_gpu_jobs = {}
    failed_jobs = []

    done_jobs = 0
    total_jobs = len(jobs)
    last_event = "Scheduler started"
    scheduler_reporting_path = jobs[0]["reporting_path"]
    scheduler_timestamp = jobs[0]["timestamp"]
    last_logged_event = None

    def record_scheduler_event(message: str, force: bool = False):
        """
        Write meaningful scheduler state changes to scheduler_live.log.

        Duplicate waiting messages are skipped so a one-second poll interval does
        not flood the log with identical lines.
        """
        nonlocal last_logged_event

        if not message:
            return

        if force or message != last_logged_event:
            append_scheduler_log(
                reporting_path=scheduler_reporting_path,
                timestamp=scheduler_timestamp,
                message=message,
            )
            last_logged_event = message

    record_scheduler_event(last_event, force=True)

    psutil.cpu_percent(interval=None)

    try:
        while pending_jobs or active_processes:
            new_failed_jobs, new_finished_jobs, finished_event = collect_finished_processes(
                active_processes=active_processes,
                active_gpu_jobs=active_gpu_jobs,
            )

            failed_jobs.extend(new_failed_jobs)
            done_jobs += len(new_finished_jobs) + len(new_failed_jobs)

            if finished_event:
                last_event = finished_event
                record_scheduler_event(last_event)

            render_dashboard(
                total_jobs=total_jobs,
                done_jobs=done_jobs,
                failed_jobs_count=len(failed_jobs),
                pending_jobs=pending_jobs,
                active_processes=active_processes,
                active_gpu_jobs=active_gpu_jobs,
                last_event=last_event,
            )

            if len(active_processes) >= max_parallel_processes:
                time.sleep(poll_seconds)
                continue

            cpu_usage = psutil.cpu_percent(interval=0.5)
            ram_usage = psutil.virtual_memory().percent

            if cpu_usage > max_cpu_usage or ram_usage > max_ram_usage:
                last_event = (
                    f"Waiting for CPU/RAM capacity | "
                    f"CPU={cpu_usage:.1f}%/{max_cpu_usage:.1f}% | "
                    f"RAM={ram_usage:.1f}%/{max_ram_usage:.1f}%"
                )
                record_scheduler_event(last_event)

                render_dashboard(
                    total_jobs=total_jobs,
                    done_jobs=done_jobs,
                    failed_jobs_count=len(failed_jobs),
                    pending_jobs=pending_jobs,
                    active_processes=active_processes,
                    active_gpu_jobs=active_gpu_jobs,
                    last_event=last_event,
                )

                time.sleep(poll_seconds)
                continue

            started_any = False

            for _ in range(len(pending_jobs)):
                if len(active_processes) >= max_parallel_processes:
                    break

                job = pending_jobs.popleft()

                needs_gpu = job.get("needs_gpu", False)
                gpu_id = None

                if needs_gpu:
                    gpu_id = choose_gpu_for_new_process(
                        max_gpu_usage=max_gpu_usage,
                        max_gpu_memory_usage=max_gpu_memory_usage,
                        active_gpu_jobs=active_gpu_jobs,
                        max_jobs_per_gpu=max_jobs_per_gpu,
                    )

                    if gpu_id is None:
                        pending_jobs.append(job)
                        continue

                process = ctx.Process(
                    target=run_single_experiment_process_entry,
                    args=(
                        job["experiment_id"],
                        job["total_experiments"],
                        job["experiment_name"],
                        job["cfg"],
                        job["run_id"],
                        job["timestamp"],
                        job["reporting_path"],
                        gpu_id,
                    ),
                )

                start_process_with_gpu_env(process, gpu_id)

                active_processes[process.pid] = {
                    "process": process,
                    "gpu_id": gpu_id,
                    "job": job,
                }

                if gpu_id is not None:
                    active_gpu_jobs[gpu_id] = active_gpu_jobs.get(gpu_id, 0) + 1

                run_id_label = job.get("run_id_label", f"run_{job['run_id']}")

                last_event = (
                    f"Started {job.get('training_type')} | "
                    f"{job['experiment_name']} | "
                    f"{run_id_label} | "
                    f"freq={job.get('freq', 'NA')} | "
                    f"noise={job.get('noise', 'NA')} | "
                    f"magnitude={job.get('magnitude', 'NA')} | "
                    f"lead_time={job.get('lead_time', 'NA')} | "
                    f"PID {process.pid} | "
                    f"GPU {gpu_id}"
                )
                record_scheduler_event(last_event)

                started_any = True

                render_dashboard(
                    total_jobs=total_jobs,
                    done_jobs=done_jobs,
                    failed_jobs_count=len(failed_jobs),
                    pending_jobs=pending_jobs,
                    active_processes=active_processes,
                    active_gpu_jobs=active_gpu_jobs,
                    last_event=last_event,
                )

                time.sleep(3)

                cpu_usage = psutil.cpu_percent(interval=0.1)
                ram_usage = psutil.virtual_memory().percent

                if cpu_usage > max_cpu_usage or ram_usage > max_ram_usage:
                    break

            if not started_any:
                if not pending_jobs:
                    last_event = (
                        f"No pending jobs left. Waiting for "
                        f"{len(active_processes)} active process(es) to exit."
                    )
                else:
                    last_event = (
                        f"Waiting for available CPU/GPU capacity. "
                        f"waiting={len(pending_jobs)}, "
                        f"running={len(active_processes)}"
                    )

                record_scheduler_event(last_event)

                render_dashboard(
                    total_jobs=total_jobs,
                    done_jobs=done_jobs,
                    failed_jobs_count=len(failed_jobs),
                    pending_jobs=pending_jobs,
                    active_processes=active_processes,
                    active_gpu_jobs=active_gpu_jobs,
                    last_event=last_event,
                )

                time.sleep(poll_seconds)

        last_event = "All jobs finished."
        record_scheduler_event(last_event, force=True)

        render_dashboard(
            total_jobs=total_jobs,
            done_jobs=done_jobs,
            failed_jobs_count=len(failed_jobs),
            pending_jobs=pending_jobs,
            active_processes=active_processes,
            active_gpu_jobs=active_gpu_jobs,
            last_event=last_event,
        )

        if failed_jobs:
            failed_names = []

            for info in failed_jobs:
                job = info["job"]
                run_id_label = job.get("run_id_label", f"run_{job['run_id']}")

                failed_names.append(
                    f"{job['experiment_name']} "
                    f"{run_id_label} "
                    f"freq={job.get('freq', 'NA')} "
                    f"noise={job.get('noise', 'NA')} "
                    f"magnitude={job.get('magnitude', 'NA')} "
                    f"lead_time={job.get('lead_time', 'NA')} "
                    f"exitcode={info['process'].exitcode}"
                )

            failure_summary = f"{len(failed_jobs)} experiment job(s) failed: {failed_names}"
            record_scheduler_event(failure_summary, force=True)

            raise RuntimeError(failure_summary)

    except KeyboardInterrupt:
        terminate_all_active_processes(active_processes)
        raise


# =============================================================================
# Missing experiment config support
# =============================================================================

# -----------------------------------------------------------------------------
# Edit only these settings.
# -----------------------------------------------------------------------------
# True  -> run only experiment_name + run_ids listed in missing_experiments.yaml
# False -> original behavior: run all experiments from all_experiments.yaml
USE_MISSING_CONFIG = False

# This file determines exactly which experiments and which run IDs are rerun.
# Expected format:
#
# source_config: all_experiments.yaml
# reporting_timestamp: 2026_06_11_RERUN_MISSING
# experiments:
#   - name: timesfm_zero_shot_026
#     run_ids: [3, 4, 5, 6, 8, 9]
MISSING_CONFIG_FILENAME = "missing_experiments.yaml"

# Used only when USE_MISSING_CONFIG = False, or as a fallback.
# FULL_CONFIG_FILENAME = "experiment_config_weekly_periods_no_chronos_timesfm.yaml"
FULL_CONFIG_FILENAME = "real_world_usda_4datasets_lt123.yaml"
FULL_CONFIG_FILENAME = "real_world_4datasets_cleaned_659_config.yaml"
FULL_CONFIG_FILENAME = "real_world_usda_config_dynamic_supplier_allocation.yaml"
FULL_CONFIG_FILENAME = "combined_experiment_config_REALWORLD_5RUNS.yaml"
FULL_CONFIG_FILENAME = "combined_experiment_config_without_chronos_timesfm_runs1_start0.yaml"
# FULL_CONFIG_FILENAME = "combined_experiment_config_only_chronos_timesfm.yaml"
# FULL_CONFIG_FILENAME = "config_local_patchtst_timemixer.yaml"
# FULL_CONFIG_FILENAME = "./missing_run_configs_6_lambda_1_tau_0/02_missing_chronos_full.yaml"
# FULL_CONFIG_FILENAME = "./missing_run_configs_6_lambda_1_tau_0/01_missing_timesfm_full.yaml"
# FULL_CONFIG_FILENAME = "./missing_run_configs_6_lambda_1_tau_0/03_other_part_1_NORMALMODE_FIXED_V2.yaml"
# FULL_CONFIG_FILENAME = "./missing_run_configs_6_lambda_1_tau_0/04_other_part_2_NORMALMODE_FIXED_V2.yaml"
# FULL_CONFIG_FILENAME = "./missing_run_configs_6_lambda_1_tau_0/05_other_part_3_NORMALMODE_FIXED_V2.yaml"
# FULL_CONFIG_FILENAME = "./missing_run_configs_6_lambda_1_tau_0/06_other_part_4_NORMALMODE_FIXED_V2.yaml"
# FULL_CONFIG_FILENAME = "combined_experiment_config.yaml"
# FULL_CONFIG_FILENAME = "combined_experiment_config_only_timesfm.yaml"
FULL_CONFIG_FILENAME = "combined_experiment_config_without_chronos_timesfm.yaml"
# FULL_CONFIG_FILENAME = "timesfm_failures_rerun.yaml"
# FULL_CONFIG_FILENAME = "failed_runs_only.yaml"

# None means:
#   - if USE_MISSING_CONFIG=True and missing_experiments.yaml contains
#     reporting_timestamp, use that value
#   - otherwise create a fresh timestamp from the current datetime
# You can also set this manually, for example:
# REPORTING_TIMESTAMP_OVERRIDE = "2026_06_11_RERUN_MISSING"
REPORTING_TIMESTAMP_OVERRIDE = None

# Base Reporting directory.
REPORTING_FOLDER_NAME = "Reporting"

# Safety switch.
# True  -> only print selected jobs, do not start experiments
# False -> actually run selected jobs
DRY_RUN = False

# Scheduler limits.
MAX_PARALLEL_PROCESSES = 50
MAX_CPU_USAGE = 85.0
MAX_RAM_USAGE = 85.0
MAX_GPU_USAGE = 70.0
MAX_GPU_MEMORY_USAGE = 0.85
MAX_JOBS_PER_GPU = 4
POLL_SECONDS = 1.0


def normalize_run_ids(run_ids_raw) -> list[int]:
    """
    Normalize run_ids from YAML.

    Supports:
        run_ids: [0, 1, 5]
        run_ids: "0, 1, 5"
        run_ids: "run_0, run_1"
    """
    if run_ids_raw is None:
        return []

    if isinstance(run_ids_raw, list):
        return sorted({int(x) for x in run_ids_raw})

    if isinstance(run_ids_raw, tuple):
        return sorted({int(x) for x in run_ids_raw})

    text = str(run_ids_raw).strip()

    if not text:
        return []

    import re

    return sorted({int(x) for x in re.findall(r"\d+", text)})


def load_missing_experiments_config(
    missing_config_path: Path,
    script_directory: Path,
) -> tuple[Path, str | None, dict[str, list[int]]]:
    """
    Load missing_experiments.yaml.

    Expected format:

        source_config: all_experiments.yaml
        reporting_timestamp: 2026_06_11_RERUN_MISSING

        experiments:
          - name: timesfm_zero_shot_022
            run_ids: [0, 4, 9]

    Returns:
        source_config_path
        reporting_timestamp
        missing_run_ids_by_experiment
    """
    missing_config_path = Path(missing_config_path)

    if not missing_config_path.exists():
        raise FileNotFoundError(
            f"Missing config does not exist: {missing_config_path}"
        )

    with missing_config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Missing config must be a YAML dictionary: {missing_config_path}"
        )

    source_config_raw = raw.get("source_config", FULL_CONFIG_FILENAME)
    source_config_path = Path(source_config_raw)

    if not source_config_path.is_absolute():
        source_config_path = script_directory / source_config_path

    reporting_timestamp = raw.get("reporting_timestamp", None)

    experiments = raw.get("experiments", [])

    if not isinstance(experiments, list) or not experiments:
        raise ValueError(
            "Missing config must contain a non-empty 'experiments' list."
        )

    missing_by_experiment: dict[str, list[int]] = {}

    for item in experiments:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid experiment entry: {item!r}")

        experiment_name = str(item.get("name", "")).strip()

        if not experiment_name:
            raise ValueError(f"Experiment entry without name: {item!r}")

        run_ids = normalize_run_ids(item.get("run_ids"))

        if not run_ids:
            raise ValueError(
                f"Experiment {experiment_name!r} has no run_ids."
            )

        if experiment_name in missing_by_experiment:
            merged = set(missing_by_experiment[experiment_name])
            merged.update(run_ids)
            missing_by_experiment[experiment_name] = sorted(merged)
        else:
            missing_by_experiment[experiment_name] = run_ids

    return source_config_path, reporting_timestamp, missing_by_experiment


def build_selected_jobs(
    experiment_configs,
    missing_by_experiment: dict[str, list[int]] | None,
    timestamp: str,
    reporting_path: Path,
) -> list[dict]:
    """
    Build jobs either for all runs, or for exact run_ids from missing config.

    If missing_by_experiment is None:
        original behavior:
            run_id = run_start_id + run_offset
            for run_offset in range(simulation_runs)

    If missing_by_experiment is given:
        rerun behavior:
            use exactly the run_ids from missing_experiments.yaml
    """
    jobs = []

    experiment_names_in_yaml = {
        experiment_name
        for experiment_name, _cfg in experiment_configs
    }

    if missing_by_experiment is not None:
        requested_names = set(missing_by_experiment.keys())
        missing_names = sorted(requested_names - experiment_names_in_yaml)

        if missing_names:
            raise ValueError(
                "These experiments are requested in missing_experiments.yaml "
                "but not found in the current source config:\n"
                + "\n".join(f"  - {name}" for name in missing_names)
            )

    for experiment_id, (experiment_name, cfg) in enumerate(experiment_configs):
        sim_runs = int(cfg["sim"]["simulation_runs"])
        run_start_id = get_run_start_id(cfg, default=0)
        training_type = cfg["sim"].get("training_type", None)
        needs_gpu = cfg_needs_gpu(cfg)
        dashboard_values = get_experiment_dashboard_values(cfg)

        if missing_by_experiment is None:
            # Original full-run mode.
            run_ids = [
                run_start_id + run_offset
                for run_offset in range(sim_runs)
            ]
        else:
            # Missing-only mode.
            # Skip every experiment not explicitly listed in missing_experiments.yaml.
            if experiment_name not in missing_by_experiment:
                continue

            # Use exactly these run IDs, no range expansion.
            run_ids = sorted(missing_by_experiment[experiment_name])

        valid_run_ids = {
            run_start_id + run_offset
            for run_offset in range(sim_runs)
        }

        for run_id in run_ids:
            if missing_by_experiment is not None and run_id not in valid_run_ids:
                raise ValueError(
                    f"Requested {experiment_name} run_{run_id}, but this run_id "
                    f"is outside the configured range from the source config. "
                    f"Configured valid run_ids are: {sorted(valid_run_ids)}"
                )

            run_offset = run_id - run_start_id
            run_id_label = f"run_{run_id}"

            jobs.append(
                {
                    "experiment_id": experiment_id,
                    "total_experiments": len(experiment_configs),
                    "experiment_name": experiment_name,
                    "cfg": cfg,

                    # Keep this numeric.
                    # It is used for seeds and Reporting.
                    "run_id": run_id,

                    # Keep the offset too, in case you need it later
                    # for debugging or dashboard extensions.
                    "run_offset": run_offset,

                    # Use this for CLI/dashboard display.
                    "run_id_label": run_id_label,

                    "timestamp": timestamp,
                    "reporting_path": reporting_path,
                    "training_type": training_type,
                    "needs_gpu": needs_gpu,

                    # Dashboard fields.
                    "freq": dashboard_values["freq"],
                    "noise": dashboard_values["noise"],
                    "magnitude": dashboard_values["magnitude"],
                    "lead_time": dashboard_values["lead_time"],
                    "dataset": dashboard_values.get("dataset", "NA"),
                    "data_source": dashboard_values.get("data_source", "NA"),
                }
            )

    return jobs


def print_job_selection_summary(
    jobs: list[dict],
    config_path: Path,
    reporting_path: Path,
    timestamp: str,
    missing_by_experiment: dict[str, list[int]] | None,
):
    print()
    print("=" * 110)
    print("SCHEDULER JOB SELECTION")
    print("=" * 110)
    print(f"Config: {config_path}")
    print(f"Reporting path: {reporting_path}")
    print(f"Timestamp: {timestamp}")
    print(f"Reporting folder: {Path(reporting_path) / str(timestamp)}")
    print(f"Jobs selected: {len(jobs)}")

    if missing_by_experiment is not None:
        requested_runs = sum(len(v) for v in missing_by_experiment.values())
        print(f"Missing-config experiments: {len(missing_by_experiment)}")
        print(f"Missing-config requested runs: {requested_runs}")

        if requested_runs != len(jobs):
            print()
            print("WARNING:")
            print(
                f"  missing_experiments.yaml requested {requested_runs} runs, "
                f"but {len(jobs)} jobs were selected."
            )
            print(
                "  This usually means some requested experiments were not found "
                "or some run_ids were invalid."
            )

    print("-" * 110)

    by_training_type = {}
    by_configuration = {}

    for job in jobs:
        training_type = str(job.get("training_type"))
        by_training_type[training_type] = by_training_type.get(training_type, 0) + 1

        configuration_key = (
            training_type,
            str(job.get("freq", "NA")),
            str(job.get("noise", "NA")),
            str(job.get("magnitude", "NA")),
            str(job.get("lead_time", "NA")),
        )
        by_configuration[configuration_key] = by_configuration.get(configuration_key, 0) + 1

    print("Runs by training type:")
    for training_type in sorted(by_training_type):
        print(f"  {training_type}: {by_training_type[training_type]} runs")

    print("-")
    print("Runs by configuration:")
    for (training_type, freq, noise, magnitude, lead_time), count in sorted(by_configuration.items()):
        print(
            f"  {training_type} | "
            f"freq={freq} | "
            f"noise={noise} | "
            f"magnitude={magnitude} | "
            f"lead_time={lead_time}: "
            f"{count} runs"
        )

    print("=" * 110)
    print()


def print_dry_run_jobs(jobs: list[dict]):
    print("DRY RUN ONLY. These jobs would be started:")
    print("-" * 110)

    for job in jobs:
        print(
            f"{job.get('training_type')} | "
            f"{job['experiment_name']} | "
            f"{job['run_id_label']} | "
            f"freq={job.get('freq', 'NA')} | "
            f"noise={job.get('noise', 'NA')} | "
            f"magnitude={job.get('magnitude', 'NA')} | "
            f"lead_time={job.get('lead_time', 'NA')} | "
            f"dataset={job.get('dataset', 'NA')} | "
            f"needs_gpu={job.get('needs_gpu')}"
        )

    print("-" * 110)
    print(f"Total dry-run jobs: {len(jobs)}")


# =============================================================================
# Entry point
# =============================================================================

def parse_cli_args(argv=None):
    """
    Parse command-line arguments.

    Examples:
        Local fixed GPUs:
            python main_slurm.py --gpu-allocation fixed --fixed-gpu-ids 0,1,2

        Slurm allocation:
            srun python main_slurm.py --gpu-allocation slurm

        Automatic mode:
            python main_slurm.py --gpu-allocation auto --fixed-gpu-ids 0,1,2
    """
    parser = argparse.ArgumentParser(
        description="Run the experiment scheduler."
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Optional prefix for the top-level Reporting folder. "
            "Example: --name 'my experiment' creates "
            "Reporting/my_experiment_<timestamp>/"
        ),
    )

    parser.add_argument(
        "--gpu-allocation",
        choices=("fixed", "slurm", "auto"),
        default="fixed",
        help=(
            "GPU allocation strategy. 'fixed' uses --fixed-gpu-ids; "
            "'slurm' uses the current Slurm allocation; 'auto' chooses Slurm "
            "inside a Slurm job and fixed IDs otherwise. Default: fixed."
        ),
    )

    parser.add_argument(
        "--fixed-gpu-ids",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help=(
            "Comma-separated GPU indexes/UUIDs for fixed or auto-local mode. "
            "Default: 0,1,2,3,4,5,6,7."
        ),
    )

    return parser.parse_args(argv)

def build_reporting_timestamp(timestamp: str, run_name: str | None) -> str:
    """
    Add the optional CLI name to the highest-level Reporting folder only.

    With --name myexperiment:
        Reporting/myexperiment_<timestamp>/<experiment_name_from_config>/run_0/...

    Without --name:
        Reporting/<timestamp>/<experiment_name_from_config>/run_0/...
    """
    if run_name is None:
        return str(timestamp)

    safe_name = safe_experiment_name(str(run_name).strip()).strip("_.-")

    if not safe_name:
        return str(timestamp)

    return f"{safe_name}_{timestamp}"


def run():
    args = parse_cli_args()

    configure_gpu_allocation(
        mode=args.gpu_allocation,
        fixed_gpu_ids=args.fixed_gpu_ids,
    )

    script_directory = Path(__file__).parent

    missing_by_experiment = None
    missing_timestamp = None

    if USE_MISSING_CONFIG:
        missing_config_path = script_directory / MISSING_CONFIG_FILENAME

        config_path, missing_timestamp, missing_by_experiment = load_missing_experiments_config(
            missing_config_path=missing_config_path,
            script_directory=script_directory,
        )
    else:
        config_path = script_directory / FULL_CONFIG_FILENAME

    experiment_configs = build_experiment_configs(config_path)

    if REPORTING_TIMESTAMP_OVERRIDE is not None:
        timestamp = str(REPORTING_TIMESTAMP_OVERRIDE)
    elif missing_timestamp is not None:
        timestamp = str(missing_timestamp)
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    timestamp = build_reporting_timestamp(
        timestamp=timestamp,
        run_name=args.name,
    )

    reporting_path = script_directory / REPORTING_FOLDER_NAME
    reporting_path.mkdir(parents=True, exist_ok=True)

    print("GPU allocation mode:", GPU_ALLOCATION_MODE)
    print("GPU slots:")
    for slot in GPU_SLOTS:
        print(
            f"  logical {slot['gpu_id']} -> "
            f"CUDA token {slot['cuda_token']} "
            f"(monitor token {slot['monitor_token']})"
        )
    print()

    jobs = build_selected_jobs(
        experiment_configs=experiment_configs,
        missing_by_experiment=missing_by_experiment,
        timestamp=timestamp,
        reporting_path=reporting_path,
    )

    print_job_selection_summary(
        jobs=jobs,
        config_path=config_path,
        reporting_path=reporting_path,
        timestamp=timestamp,
        missing_by_experiment=missing_by_experiment,
    )

    if not jobs:
        raise RuntimeError("No jobs selected.")

    if DRY_RUN:
        print_dry_run_jobs(jobs)
        print()
        print("DRY_RUN is currently True.")
        print("After checking the selected jobs, set DRY_RUN = False and run again.")
        return

    run_load_balanced(
        jobs=jobs,

        # Total number of simultaneous experiment processes.
        max_parallel_processes=MAX_PARALLEL_PROCESSES,

        # New jobs only start if CPU/RAM are below these thresholds.
        max_cpu_usage=MAX_CPU_USAGE,
        max_ram_usage=MAX_RAM_USAGE,

        # New GPU jobs only start if a GPU is below this utilization threshold.
        max_gpu_usage=MAX_GPU_USAGE,

        # New GPU jobs only start if GPU memory usage is below this fraction.
        max_gpu_memory_usage=MAX_GPU_MEMORY_USAGE,

        # With 3 GPUs and max_jobs_per_gpu=15, this allows up to 45 GPU jobs.
        # Reduce this for heavy models.
        max_jobs_per_gpu=MAX_JOBS_PER_GPU,

        # Scheduler check interval.
        poll_seconds=POLL_SECONDS,
    )


if __name__ == "__main__":
    run()
