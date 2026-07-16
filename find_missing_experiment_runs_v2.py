#!/usr/bin/env python3
"""
Scan an experiment batch directory and generate recovery YAML files.

Expected directory layout:
    PARENT/
      <experiment_name>/
        run_0/
        run_1/
        ...
        run_9/

Default output (written below --output-dir):
1. missing_experiments.yaml
   - Full base configuration
   - Only incomplete/missing experiments
   - Explicit missing_run_ids metadata per experiment
   - Recovery summary and naming validation

Optional output with --write-rerun-configs:
2. rerun_configs/rerun_run_<ID>.yaml
   - One runner-compatible YAML per missing run ID
   - sim.simulation_runs = 1
   - sim.run_start_id = <ID>
   - Only experiments missing that run
   - Per-experiment run_start_id overrides are removed so the global run ID wins

Requires:
    pip install pyyaml
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it with: pip install pyyaml"
    ) from exc


SEASONALITY_LABELS = {
    52: "yearly",
    26: "half_yearly",
    13: "quarterly",
    7: "half_quarterly",
}

MAGNITUDE_LABELS = {
    50: "low_mag",
    100: "medium_mag",
    200: "high_mag",
}

KNOWN_MODEL_ALIASES = {
    "chronos_zero_shot": "chronos_zero_shot",
    "chronos": "chronos_zero_shot",
    "timesfm_zero_shot": "timesfm_zero_shot",
    "timesfm": "timesfm_zero_shot",
    "local_patch": "local_patchtst",
    "local_patchtst": "local_patchtst",
    "split_patch": "split_patchtst",
    "split_patchtst": "split_patchtst",
    "local_timemixer": "local_timemixer",
    "split_timemixer": "split_timemixer",
    "local_multichannel": "local_multichannel",
    "split_multichannel": "split_multichannel",
    "no_training": "no_training_ma",
    "no_training_ma": "no_training_ma",
    "ma": "no_training_ma",
}

EXPERIMENT_NAME_RE = re.compile(
    r"^(?P<model>.+?)_(?P<index>\d{3})_"
    r"(?P<descriptor>"
    r"(?:yearly|half_yearly|quarterly|half_quarterly)_"
    r"(?:low_mag|medium_mag|high_mag)_"
    r"noise[^_]+_lt[^_]+"
    r")$"
)

RUN_DIR_RE = re.compile(r"^run_(\d+)$")


class ConfigError(RuntimeError):
    """Raised when the source configuration has an unexpected structure."""


def parse_run_ids(value: str) -> list[int]:
    """Parse values such as '0-9', '0,2,5-7', or '3'."""
    result: set[int] = set()

    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid run range: {part!r}"
                ) from exc

            if start < 0 or end < 0 or end < start:
                raise argparse.ArgumentTypeError(
                    f"Invalid run range: {part!r}"
                )
            result.update(range(start, end + 1))
        else:
            try:
                run_id = int(part)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid run ID: {part!r}"
                ) from exc

            if run_id < 0:
                raise argparse.ArgumentTypeError(
                    f"Run IDs must be non-negative: {part!r}"
                )
            result.add(run_id)

    if not result:
        raise argparse.ArgumentTypeError("At least one run ID is required.")

    return sorted(result)


def normalize_model_name(value: str) -> str:
    """Normalize user-facing model aliases."""
    key = value.strip().lower().replace("-", "_")
    return KNOWN_MODEL_ALIASES.get(key, key)


def split_model_arguments(values: Iterable[str] | None) -> list[str]:
    """Allow '--models a b' and '--models a,b'."""
    if not values:
        return []

    models: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                models.append(normalize_model_name(item))
    return list(dict.fromkeys(models))


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError("The YAML root must be a mapping.")
    if not isinstance(loaded.get("experiments"), list):
        raise ConfigError("The YAML must contain an 'experiments' list.")

    return loaded


def nested_get(mapping: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def effective_value(
    experiment: dict[str, Any],
    base_config: dict[str, Any],
    path: tuple[str, ...],
    default: Any = None,
) -> Any:
    override = nested_get(experiment.get("overrides", {}), path, default=None)
    if override is not None:
        return override
    return nested_get(base_config, path, default=default)


def model_from_experiment(experiment: dict[str, Any]) -> str:
    name = str(experiment.get("name", ""))
    match = EXPERIMENT_NAME_RE.match(name)
    if match:
        return normalize_model_name(match.group("model"))

    training_type = nested_get(
        experiment.get("overrides", {}),
        ("sim", "training_type"),
        default=None,
    )
    if training_type is None:
        return "no_training_ma"
    return normalize_model_name(str(training_type))


def safe_token(value: Any) -> str:
    token = str(value).strip().lower()
    token = token.replace(".", "p").replace("-", "minus")
    return re.sub(r"[^a-z0-9]+", "_", token).strip("_")


def expected_descriptor(
    experiment: dict[str, Any],
    base_config: dict[str, Any],
) -> str:
    frequency = effective_value(
        experiment,
        base_config,
        ("market", "seasonality_frequncy"),
    )
    magnitude = effective_value(
        experiment,
        base_config,
        ("market", "seasonality_magnitude"),
    )
    noise = effective_value(
        experiment,
        base_config,
        ("market", "random_walk", "mean"),
    )

    seasonality = SEASONALITY_LABELS.get(
        frequency,
        f"weeks_{safe_token(frequency)}",
    )
    mag_label = MAGNITUDE_LABELS.get(
        magnitude,
        f"mag_{safe_token(magnitude)}",
    )
    noise_label = f"noise{safe_token(noise)}"

    level_0 = effective_value(
        experiment,
        base_config,
        ("supply_chain", "sc_levels", "sc_level_0", "lead_time"),
    )
    level_1 = effective_value(
        experiment,
        base_config,
        ("supply_chain", "sc_levels", "sc_level_1", "lead_time"),
    )

    if level_0 == level_1:
        lead_time_label = f"lt{safe_token(level_0)}"
    else:
        lead_time_label = (
            f"ltmixed_{safe_token(level_0)}_{safe_token(level_1)}"
        )

    return f"{seasonality}_{mag_label}_{noise_label}_{lead_time_label}"


def validate_experiment_name(
    experiment: dict[str, Any],
    base_config: dict[str, Any],
) -> dict[str, Any] | None:
    name = str(experiment.get("name", ""))
    match = EXPERIMENT_NAME_RE.match(name)

    if not match:
        return {
            "name": name,
            "issue": "name_does_not_match_expected_pattern",
        }

    actual = match.group("descriptor")
    expected = expected_descriptor(experiment, base_config)
    if actual != expected:
        return {
            "name": name,
            "issue": "descriptor_does_not_match_overrides",
            "actual_descriptor": actual,
            "expected_descriptor": expected,
        }

    return None


def discover_models(
    parent: Path,
    experiments_by_name: dict[str, dict[str, Any]],
) -> list[str]:
    """Detect models from experiment folders already present in the parent."""
    detected: set[str] = set()

    for child in parent.iterdir():
        if not child.is_dir():
            continue
        experiment = experiments_by_name.get(child.name)
        if experiment is not None:
            detected.add(model_from_experiment(experiment))

    return sorted(detected)


def run_ids_in_directory(experiment_dir: Path) -> tuple[list[int], list[str]]:
    present: set[int] = set()
    unexpected: list[str] = []

    if not experiment_dir.is_dir():
        return [], unexpected

    for child in experiment_dir.iterdir():
        if not child.is_dir():
            continue
        match = RUN_DIR_RE.match(child.name)
        if match:
            present.add(int(match.group(1)))
        else:
            unexpected.append(child.name)

    return sorted(present), sorted(unexpected)


def remove_experiment_run_controls(experiment: dict[str, Any]) -> dict[str, Any]:
    """Ensure the generated per-run global sim settings are authoritative."""
    cleaned = copy.deepcopy(experiment)
    overrides = cleaned.get("overrides")
    if not isinstance(overrides, dict):
        return cleaned

    sim_overrides = overrides.get("sim")
    if isinstance(sim_overrides, dict):
        sim_overrides.pop("run_start_id", None)
        sim_overrides.pop("simulation_runs", None)

    return cleaned


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find missing experiment directories and missing run_<ID> "
            "subdirectories, then create recovery YAML files."
        )
    )
    parser.add_argument(
        "--parent",
        required=True,
        type=Path,
        help="Batch parent directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Master YAML containing the complete experiments list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for all generated YAML files. "
            "Default: the --parent directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("missing_experiments.yaml"),
        help=(
            "Filename of the recovery index YAML, relative to --output-dir, "
            "or an absolute file path. Default: missing_experiments.yaml"
        ),
    )
    parser.add_argument(
        "--rerun-dir",
        type=Path,
        default=Path("rerun_configs"),
        help=(
            "Subdirectory for optional per-run YAML files. This is used only "
            "with --write-rerun-configs. The path is relative to --output-dir "
            "unless it is absolute. Default: rerun_configs"
        ),
    )
    parser.add_argument(
        "--run-ids",
        type=parse_run_ids,
        default=parse_run_ids("0-9"),
        help="Expected IDs, e.g. '0-9' or '0,2,5-7'. Default: 0-9",
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--models",
        nargs="+",
        help=(
            "Only scan selected model groups, e.g. "
            "'--models local_patchtst local_timemixer'. "
            "Comma-separated values and aliases such as local_patch are accepted."
        ),
    )
    selection.add_argument(
        "--all-models",
        action="store_true",
        help="Scan all model groups in the master YAML.",
    )

    parser.add_argument(
        "--write-rerun-configs",
        action="store_true",
        help=(
            "Additionally write one executable YAML per missing run ID. "
            "Without this flag, exactly one YAML is written."
        ),
    )
    parser.add_argument(
        "--strict-names",
        action="store_true",
        help="Exit with an error when experiment names disagree with their overrides.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    parent = args.parent.expanduser().resolve()
    config_path = args.config.expanduser().resolve()

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else parent
    )
    output_arg = args.output.expanduser()
    rerun_arg = args.rerun_dir.expanduser()
    output_path = (
        output_arg.resolve()
        if output_arg.is_absolute()
        else (output_root / output_arg).resolve()
    )
    rerun_dir = (
        rerun_arg.resolve()
        if rerun_arg.is_absolute()
        else (output_root / rerun_arg).resolve()
    )

    if not parent.is_dir():
        raise ConfigError(f"Parent directory does not exist: {parent}")

    base_config = load_yaml(config_path)
    all_experiments = base_config["experiments"]

    experiments_by_name: dict[str, dict[str, Any]] = {}
    duplicate_names: list[str] = []
    for experiment in all_experiments:
        if not isinstance(experiment, dict) or not experiment.get("name"):
            raise ConfigError("Every experiment must be a mapping with a non-empty name.")
        name = str(experiment["name"])
        if name in experiments_by_name:
            duplicate_names.append(name)
        experiments_by_name[name] = experiment

    if duplicate_names:
        duplicates = ", ".join(sorted(set(duplicate_names)))
        raise ConfigError(f"Duplicate experiment names in YAML: {duplicates}")

    available_models = sorted(
        {model_from_experiment(experiment) for experiment in all_experiments}
    )

    requested_models = split_model_arguments(args.models)
    if args.all_models:
        selected_models = available_models
        selection_mode = "all_models"
    elif requested_models:
        unknown = sorted(set(requested_models) - set(available_models))
        if unknown:
            raise ConfigError(
                "Unknown model group(s): "
                + ", ".join(unknown)
                + ". Available: "
                + ", ".join(available_models)
            )
        selected_models = requested_models
        selection_mode = "explicit"
    else:
        selected_models = discover_models(parent, experiments_by_name)
        selection_mode = "auto_detected_from_existing_folders"
        if not selected_models:
            raise ConfigError(
                "No experiment folders from the YAML were found in the parent. "
                "Specify --models ... or use --all-models."
            )

    selected_experiments = [
        experiment
        for experiment in all_experiments
        if model_from_experiment(experiment) in set(selected_models)
    ]

    name_warnings = [
        warning
        for experiment in selected_experiments
        if (warning := validate_experiment_name(experiment, base_config)) is not None
    ]
    if args.strict_names and name_warnings:
        raise ConfigError(
            f"{len(name_warnings)} experiment name(s) disagree with their overrides. "
            "Run without --strict-names to include warnings in the report."
        )

    expected_run_ids = args.run_ids
    expected_run_set = set(expected_run_ids)

    missing_entries: list[dict[str, Any]] = []
    complete_count = 0
    missing_directory_count = 0
    missing_run_count = 0

    selected_names = {str(exp["name"]) for exp in selected_experiments}

    ignored_parent_dir_names: set[str] = set()
    for generated_path in (output_path.parent, rerun_dir):
        try:
            relative = generated_path.relative_to(parent)
        except ValueError:
            continue
        if relative.parts:
            ignored_parent_dir_names.add(relative.parts[0])

    unexpected_experiment_dirs = sorted(
        child.name
        for child in parent.iterdir()
        if child.is_dir()
        and child.name not in selected_names
        and child.name not in ignored_parent_dir_names
    )

    for experiment in selected_experiments:
        name = str(experiment["name"])
        experiment_dir = parent / name
        present_all, unexpected_run_dirs = run_ids_in_directory(experiment_dir)
        present_expected = sorted(expected_run_set.intersection(present_all))
        missing = sorted(expected_run_set.difference(present_all))
        extra_run_ids = sorted(set(present_all).difference(expected_run_set))

        if not missing:
            complete_count += 1
            continue

        if not experiment_dir.is_dir():
            missing_directory_count += 1
            status = "missing_experiment_directory"
        else:
            status = "missing_runs"

        missing_run_count += len(missing)

        report_experiment = copy.deepcopy(experiment)
        report_experiment["missing_run_ids"] = missing
        report_experiment["present_run_ids"] = present_expected
        report_experiment["folder_status"] = status
        report_experiment["expected_folder"] = str(experiment_dir)
        if extra_run_ids:
            report_experiment["extra_run_ids"] = extra_run_ids
        if unexpected_run_dirs:
            report_experiment["unexpected_subdirectories"] = unexpected_run_dirs

        missing_entries.append(report_experiment)

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    recovery_config = copy.deepcopy(base_config)
    recovery_config["experiments"] = missing_entries
    recovery_config["recovery"] = {
        "generated_at_utc": generated_at,
        "source_config": str(config_path),
        "parent_dir": str(parent),
        "output_dir": str(output_root),
        "expected_run_ids": expected_run_ids,
        "selected_models": selected_models,
        "model_selection_mode": selection_mode,
        "available_models": available_models,
        "summary": {
            "expected_experiments": len(selected_experiments),
            "complete_experiments": complete_count,
            "incomplete_experiments": len(missing_entries),
            "missing_experiment_directories": missing_directory_count,
            "missing_run_count": missing_run_count,
        },
        "naming_maps": {
            "seasonality_weeks": SEASONALITY_LABELS,
            "seasonality_magnitude": MAGNITUDE_LABELS,
            "noise": {
                "source": "market.random_walk.mean",
                "format": "noise<VALUE>",
            },
            "lead_time": {
                "source": (
                    "supply_chain.sc_levels.sc_level_0/1.lead_time"
                ),
                "format": "lt<VALUE>",
            },
        },
        "name_validation_warnings": name_warnings,
        "unexpected_parent_directories": unexpected_experiment_dirs,
    }
    dump_yaml(recovery_config, output_path)

    generated_rerun_files: list[Path] = []
    if args.write_rerun_configs:
        # Remove old rerun YAMLs generated by this script, but leave other files untouched.
        rerun_dir.mkdir(parents=True, exist_ok=True)
        for old_file in rerun_dir.glob("rerun_run_*.yaml"):
            old_file.unlink()

        for run_id in expected_run_ids:
            missing_for_run = [
                remove_experiment_run_controls(experiment)
                for experiment in selected_experiments
                if not (parent / str(experiment["name"]) / f"run_{run_id}").is_dir()
            ]
            if not missing_for_run:
                continue

            rerun_config = copy.deepcopy(base_config)
            rerun_config.setdefault("sim", {})
            rerun_config["sim"]["simulation_runs"] = 1
            rerun_config["sim"]["run_start_id"] = run_id
            rerun_config["experiments"] = missing_for_run

            rerun_path = rerun_dir / f"rerun_run_{run_id}.yaml"
            dump_yaml(rerun_config, rerun_path)
            generated_rerun_files.append(rerun_path)

    summary = recovery_config["recovery"]["summary"]
    print(f"Wrote recovery index: {output_path}")
    print(f"Selected models: {', '.join(selected_models)}")
    print(
        "Experiments: "
        f"{summary['complete_experiments']} complete, "
        f"{summary['incomplete_experiments']} incomplete"
    )
    print(f"Missing run folders: {summary['missing_run_count']}")
    if generated_rerun_files:
        print(f"Wrote {len(generated_rerun_files)} rerun config(s) to: {rerun_dir}")
    elif args.write_rerun_configs:
        print("No rerun configs were needed.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
