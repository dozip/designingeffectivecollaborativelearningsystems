import sys
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

from helpers.helpers import build_experiment_configs, reset_simulaltion_from_dict
from Simulation_Component.reporting import Reporting
from Simulation_Component.runner import run_simulation_phase
from ML_Backends import build_backend
from ML_Backends.ma import NoOpBackend

### Def Logging:
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)


def run():
    np.random.seed(42)
    random.seed(42)

    script_directory = Path(__file__).parent
    config_path = script_directory / "config.yaml"
    experiment_configs = build_experiment_configs(config_path)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    reporting_path = script_directory / "Reporting"
    reporting_path.mkdir(parents=True, exist_ok=True)

    for experiment_id, (experiment_name, cfg) in enumerate(experiment_configs):
        simulation, market, supply_chain, sc_agent_list = reset_simulaltion_from_dict(cfg)

        for run_id in range(simulation.sim_runs):
            logger.info(f"Experiment {experiment_id + 1}/{len(experiment_configs)}: {experiment_name}")
            logger.info(f"Run {run_id + 1}/{simulation.sim_runs}")

            simulation, market, supply_chain, sc_agent_list = reset_simulaltion_from_dict(cfg)
            backend = build_backend(cfg)

            test_end = simulation.conv_time + simulation.sim_time + simulation.testing_time
            warmup_end = (simulation.conv_time + simulation.sim_time
                          if backend.needs_training_phase else test_end)

            # Warm-up phase: always vanilla per-agent inference, regardless of backend
            run_simulation_phase(0, warmup_end, simulation, market,
                                 supply_chain, sc_agent_list, NoOpBackend(cfg))

            # Train (or no-op / attach)
            val_loss = backend.train(simulation, market, supply_chain, sc_agent_list)

            # Inference phase (skipped when warm-up already ran to the end)
            if warmup_end < test_end:
                run_simulation_phase(warmup_end, test_end, simulation, market,
                                     supply_chain, sc_agent_list, backend)

            Reporting(path=reporting_path,
                      timestamp=f"{timestamp}_{experiment_name}").create_reporting_multiple_runs(
                agent_list=sc_agent_list, market=market, supply_chain=supply_chain,
                cfg=cfg, run_id=run_id, val_loss=val_loss,
            )


if __name__ == "__main__":
    run()
