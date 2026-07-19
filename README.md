# Designing Effective Collaborative Learning Systems: Demand Forecasting in Supply Chains Using Distributed Data

This repository contains the code used for the paper *Designing Effective Collaborative Learning Systems: Demand Forecasting in Supply Chains Using Distributed Data*, which is currently under review.

## Setup

The experiments were conducted using Python 3.12.9.

1. Create a Python environment.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Install Jupyter Notebook to run the evaluation notebook.
4. For GPU execution, install the required CUDA-compatible versions of the machine-learning libraries.

## Configuration

The experiments are configured using the YAML configuration files in the repository, including:

- `combined_experiment_config.yaml` for synthetic experiments
- `combined_experiment_config_REALWORLD.yaml` for real-world experiments

The configuration files define the forecasting approach, simulation parameters, market configuration, supply-chain structure, and number of runs.

## Running the Experiments

The experiment scheduler is implemented in `main_slurm.py`. Before starting an experiment, select the required experiment configuration in this file.

For execution on locally available GPUs, the scheduler can be started with:

```bash
python main_slurm.py --gpu-allocation fixed --fixed-gpu-ids 0,1,2,3
```

The selected GPU IDs must match the GPUs available on the system.

### Running on a Slurm GPU Cluster

The file `main_slurm.slurm` provides an example Slurm submission script for running the experiments on multiple GPUs.

Before submission, adapt the following values to the target cluster:

- Slurm partition and requested GPU type
- number of GPUs and CPUs
- project directory
- Python environment
- job name and output paths

Submit the job with:

```bash
sbatch main_slurm.slurm
```

The Slurm script starts `main_slurm.py` using the GPU allocation assigned by Slurm. Experiment outputs, scheduler logs, and per-run logs are written to the `Reporting/` directory.

## Real-World Data

The real-world dataset used in the experiments is available in the `./data` directory.

The data was extracted through the corresponding API using:

```text
get_data.py
```

## Results

The processed results used for the study are provided in the `Results/` directory:

- `Results_real_world_data_multi_product.xlsx`
- `Results_synthethic_multi_product_lambda_075_tau_0.xlsx`
- `Results_synthethic_multi_product_lambda_075_tau_2.xlsx`
- `Results_synthethic_multi_product_lambda_1_tau_0.xlsx`
- `Results_synthethic_multi_product_lambda_1_tau_2.xlsx`

The Excel files contain aggregated evaluation results and summary tables for the real-world and synthetic experiments. The values of `lambda` and `tau` in the filenames identify the respective synthetic market configuration.

These Excel files are **not the raw simulation data**. The complete raw output of approximately 39,000 experimental runs requires too much storage to be included in this repository. Raw outputs are generated in the `Reporting/` directory when the experiments are executed.

The aggregation and evaluation of the individual runs can be reproduced with:

```text
evaluation_notebook.ipynb
```
