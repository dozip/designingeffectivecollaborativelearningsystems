# Designing Effective Collaborative Learning Systems: Demand Forecasting in Supply Chains Using Distributed Data

This code is part of the paper ***Designing Effective Collaborative Learning Systems: Demand Forecasting in Supply Chains Using Distributed Data*** which is currently under review.

## Setup
To run the code, please follow the following instruction:
1. The simulation was run using python 3.9.6
2. Use the _requirements.txt_ to install necessary packages
3. Install _jupyter-notebook_
4. Usage of NVIDA GPUs is possible, if necessary packages are installed.

### Configuration
The code can be configured using the _config.yaml_ file. Here the possible parameters can be set. The most important parameters described below. For all other parameters please refer to the paper or config-file.

### Training_type:
Sets the typ of training. 
- None: a moving average forecasting is used
- loca_multichannel: training of the architecture is done locally
 split_multichannel: training of the architecture is done collaboratively

 ### Data_scource:
 Here, we define how to simulate the market data. If _None_ is given, synthetic market data is generated based on the defined parameters. If a path is given, the data from that path is given. You can use the _FoodManuData1993_2024.xlsx_ as blue print for how to format your data. 

 ### Running the Code
 After isntalling all necessary packages and defining all parameters using the config-file, the simualtion can be started by running:
 ```bash
python main.py
```

The result will be saved in the *Reporting-Folder*. The results raw data of the paper can be found in the folder *Results*

## Results of our study
We analyzed the resutls using a juypter-note book (see name below). Make sure that you can run juypter-notebooks.
```bash
evaluation_itegrated_Experiment.ipynb
```

The raw data can be found in the folder ***Results/Rawdata/***. Experiments I to III are the results of the synthethic data for leadtime 1 to 3. Eval_Realdworld I to III are the results using real world data and leadtimes 1 to 3.

The evaulation can be found in the folder ***Results/Evaluation/***. 
