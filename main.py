import os

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

import numpy as np
import psutil
import torch

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




def get_experiment_dashboard_values(cfg: dict) -> dict:
    """
    Extract values that should be shown in the CLI dashboard.

    For your YAML:
        frequency -> market.seasonality_frequncy
        noise     -> market.random_walk.variance
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

    lead_time = get_supply_chain_lead_value(cfg, default="NA")

    return {
        "freq": compact_value_for_path(freq),
        "noise": compact_value_for_path(noise),
        "lead_time": compact_value_for_path(lead_time),
    }
def build_experiment_folder_name(experiment_name: str, cfg: dict) -> str:
    """
    Final experiment folder format:
        <experiment_name>

    Example:
        timesfm_zero_shot_001

    The frequency/noise/lead_time values are shown in the CLI dashboard,
    not duplicated in the folder name.
    """
    return safe_experiment_name(experiment_name)

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
# GPU monitoring
# =============================================================================

def get_gpu_usage():
    """
    Returns GPU utilization percentages from nvidia-smi.

    Example:
        [12.0, 63.0]
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception:
        return []

    usage = []

    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        usage.append(float(line.strip()))

    return usage


def get_gpu_memory_usage():
    """
    Returns GPU memory usage from nvidia-smi.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception:
        return []

    memory_stats = []

    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue

        used_raw, total_raw = line.split(",")

        used_mb = float(used_raw.strip())
        total_mb = float(total_raw.strip())

        memory_stats.append(
            {
                "used_mb": used_mb,
                "total_mb": total_mb,
                "memory_usage": used_mb / total_mb if total_mb > 0 else 1.0,
            }
        )

    return memory_stats


def get_gpu_stats():
    """
    Combines GPU utilization and memory usage.
    """
    gpu_usage = get_gpu_usage()

    if not gpu_usage:
        return []

    gpu_memory = get_gpu_memory_usage()

    stats = []

    for gpu_id, usage in enumerate(gpu_usage):
        if gpu_id < len(gpu_memory):
            mem = gpu_memory[gpu_id]
            memory_used_mb = mem["used_mb"]
            memory_total_mb = mem["total_mb"]
            memory_usage = mem["memory_usage"]
        else:
            memory_used_mb = 0.0
            memory_total_mb = 0.0
            memory_usage = 0.0

        stats.append(
            {
                "gpu_id": gpu_id,
                "gpu_usage": float(usage),
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "memory_usage": memory_usage,
            }
        )

    return stats


def choose_gpu_for_new_process(
    max_gpu_usage: float = 70.0,
    max_gpu_memory_usage: float = 0.85,
    active_gpu_jobs: dict | None = None,
    max_jobs_per_gpu: int = 1,
):
    """
    Choose a GPU for a new process.

    A GPU is eligible if:
    - utilization <= max_gpu_usage
    - memory usage <= max_gpu_memory_usage
    - currently scheduled jobs on that GPU < max_jobs_per_gpu
    """
    if active_gpu_jobs is None:
        active_gpu_jobs = {}

    gpu_stats = get_gpu_stats()

    if not gpu_stats:
        return None

    candidates = []

    for gpu in gpu_stats:
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
                f"lead_time={job.get('lead_time', 'NA')} | "
                f"GPU {info['gpu_id']}"
            )
    else:
        print("Currently running: none")

    print("=" * 110)
    print("Press Ctrl+C to stop.")
    sys.stdout.flush()


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

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
    """
    try:
        run_single_experiment(*args, **kwargs)

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        os._exit(0)

    except BaseException:
        try:
            logger.exception("Experiment process failed.")
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        os._exit(1)


# =============================================================================
# Process management
# =============================================================================

def start_process_with_gpu_env(process, gpu_id):
    """
    Start a child process with CUDA_VISIBLE_DEVICES set before spawn.

    This is important because the child imports torch during startup.
    """
    old_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    old_cuda_device_order = os.environ.get("CUDA_DEVICE_ORDER")

    try:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
            last_event = (
                f"FAILED {job.get('training_type')} | "
                f"{job['experiment_name']} | "
                f"{run_id_label} | "
                f"freq={job.get('freq', 'NA')} | "
                f"noise={job.get('noise', 'NA')} | "
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
                    f"lead_time={job.get('lead_time', 'NA')} | "
                    f"PID {process.pid} | "
                    f"GPU {gpu_id}"
                )

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

                time.sleep(10)

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

        render_dashboard(
            total_jobs=total_jobs,
            done_jobs=done_jobs,
            failed_jobs_count=len(failed_jobs),
            pending_jobs=pending_jobs,
            active_processes=active_processes,
            active_gpu_jobs=active_gpu_jobs,
            last_event="All jobs finished.",
        )

        if failed_jobs:
            failed_names = [
                (
                    f"{info['job']['experiment_name']} "
                    f"{info['job'].get('run_id_label', f"run_{info['job']['run_id']}")} "
                    f"freq={info['job'].get('freq', 'NA')} "
                    f"noise={info['job'].get('noise', 'NA')} "
                    f"lead_time={info['job'].get('lead_time', 'NA')} "
                    f"exitcode={info['process'].exitcode}"
                )
                for info in failed_jobs
            ]

            raise RuntimeError(
                f"{len(failed_jobs)} experiment job(s) failed: {failed_names}"
            )

    except KeyboardInterrupt:
        terminate_all_active_processes(active_processes)
        raise


# =============================================================================
# Entry point
# =============================================================================

def run():
    script_directory = Path(__file__).parent
    # config_path = script_directory / "config.yaml"
    config_path = script_directory / "final_full_factorial_config_compact_names.yaml"

    experiment_configs = build_experiment_configs(config_path)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    reporting_path = script_directory / "Reporting"
    reporting_path.mkdir(parents=True, exist_ok=True)

    jobs = []

    for experiment_id, (experiment_name, cfg) in enumerate(experiment_configs):
        sim_runs = cfg["sim"]["simulation_runs"]
        training_type = cfg["sim"].get("training_type", None)
        needs_gpu = cfg_needs_gpu(cfg)

        dashboard_values = get_experiment_dashboard_values(cfg)

        for run_number in range(sim_runs):
            run_id_label = f"run_{run_number}"

            jobs.append(
                {
                    "experiment_id": experiment_id,
                    "total_experiments": len(experiment_configs),
                    "experiment_name": experiment_name,
                    "cfg": cfg,

                    # Keep this numeric.
                    # It is used for seeds and Reporting.
                    "run_id": run_number,

                    # Use this for CLI/dashboard display.
                    "run_id_label": run_id_label,

                    "timestamp": timestamp,
                    "reporting_path": reporting_path,
                    "training_type": training_type,
                    "needs_gpu": needs_gpu,

                    # Dashboard fields.
                    "freq": dashboard_values["freq"],
                    "noise": dashboard_values["noise"],
                    "lead_time": dashboard_values["lead_time"],
                }
            )

    run_load_balanced(
        jobs=jobs,

        # Total number of simultaneous experiment processes.
        max_parallel_processes=18,

        # New jobs only start if CPU/RAM are below these thresholds.
        max_cpu_usage=75.0,
        max_ram_usage=85.0,

        # New GPU jobs only start if a GPU is below this utilization threshold.
        max_gpu_usage=70.0,

        # New GPU jobs only start if GPU memory usage is below this fraction.
        max_gpu_memory_usage=0.85,

        # With 3 GPUs and max_jobs_per_gpu=8, this allows up to 24 GPU jobs.
        # For heavy ML training, reduce this to 1.
        max_jobs_per_gpu=6,

        # Scheduler check interval.
        poll_seconds=1.0,
    )

if __name__ == "__main__":
    run()