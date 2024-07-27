import yaml
import json
import subprocess
import re

from pathlib import Path

from Simulation_Component.market import *
from Simulation_Component.agent import *
from Simulation_Component.supply_chain import *
from Simulation_Component.simulation import Simulation


def load_config(path: Path):
    """load the config for the simulation

    Args:
        path (Path): path to config file

    Returns:
        dict: containing all the configurations
    """
    with open(path, "r") as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)

    return cfg


def init_simulaltion(path: Path) -> list[Simulation, Market, Supply_Chain, list, dict]:
    """Initate all objects needed for simulations

    Args:
        path (Path):path to config file

    Returns:
        list[Simulation, Market, Supply_Chain, list, dict]: return the initialized objects Simulatio,
        Market and Supply Chain, a list of all agents ordered per supply chain level and the rest of the config file
    """

    cfg = load_config(path)

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
    market = cfg['market']
    data_source = cfg['market']['data_scource']
    primary_demand = market["primary_demand"]
    trend_magnitude = market["trend_magnitude"]
    seansonality_magnitude = market["seasonality_magnitude"]
    seasonality_frequncy = market["seasonality_frequncy"]
    random_walk = market["random_walk"]
    mean = random_walk["mean"]
    variance = random_walk["variance"]
    market_demand_split = market["demand_split"]

    sc_all = cfg['supply_chain']
    agents_per_level = sc_all['agents_per_level']
    sc_levels = sc_all['sc_levels']

    sc_agent_list = []
    sc_adjacency = []
    sc_lead_time = []

    if data_source is None:
        # Init Simulation
        simulation = Simulation(T=T, sim_runs=sim_runs, sim_time=sim_time, conv_time=conv_time, retraining_time=training_time, 
                                testing_time=testing_time, training_type=training_type, train_size=train_size, val_size=val_size)
        # Init Market
        market = MarketArtifical(sim_time=T, primary_demand=primary_demand, trend_mag=trend_magnitude,
                        seasonality_mag=seansonality_magnitude, seasonality_freq=seasonality_frequncy,
                        random_walk_mean=mean, random_walk_var=variance, retailer_num=agents_per_level[0], market_demand_split=market_demand_split)
    else:
        #load data from csv
        df = pd.read_excel(data_source) 
        data = df.iloc[0].values
        data = np.array([int(x.replace(',', '')) for x in data])/1000

        # set simulations paramters
        if len(data) < T:
            raise ValueError("Simulation time is too long for the given Dataset")
        if len(data)< training_time:
            raise ValueError("TrainingTime is is too long for the given Dataset")

        simulation = Simulation(T=T, sim_runs=sim_runs, sim_time=sim_time, conv_time=conv_time, retraining_time=training_time, 
                                testing_time=testing_time, training_type=training_type, train_size=train_size, val_size=val_size)
        
        # create market
        market = MarketDataSource(data, retailer_num=agents_per_level[0], market_demand_split=market_demand_split)

    # Init Agentes
    for i, level in enumerate(sc_levels):
        num_agents = agents_per_level[i]
        sc = sc_levels[level]
        adjaceny_matrix = np.array(sc["adjaceny_list"])
        lead_time_matrix = np.array(sc["lead_time"])
        sc_adjacency.append(adjaceny_matrix)
        sc_lead_time.append(lead_time_matrix)

        agent_list = []
        for j in range(num_agents):
            agent = Agent(id=j, sc_level=i, adjacency_matrix=adjaceny_matrix,
                          lead_time_matrix=lead_time_matrix, training_time=training_time, cfg=sc, cfg_all=sc_all,
                          sequence_length=sequence_length, eopchs=epochs, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)
            agent_list.append(agent)

        sc_agent_list.append(agent_list)
    # Init Supply Chain
    supply_chain = Supply_Chain(sc_adjacency, sc_lead_time)

    return simulation, market, supply_chain, sc_agent_list, cfg


def reset_simulaltion(path: Path) -> list[Simulation, Market, Supply_Chain, list, dict]:
    """Initate all objects needed for simulations

    Args:
        path (Path):path to config file

    Returns:
        list[Simulation, Market, Supply_Chain, list, dict]: return the initialized objects Simulatio,
        Market and Supply Chain, a list of all agents ordered per supply chain level and the rest of the config file
    """

    cfg = load_config(path)

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
    market = cfg['market']
    data_source = cfg['market']['data_scource']
    primary_demand = market["primary_demand"]
    trend_magnitude = market["trend_magnitude"]
    seansonality_magnitude = market["seasonality_magnitude"]
    seasonality_frequncy = market["seasonality_frequncy"]
    random_walk = market["random_walk"]
    mean = random_walk["mean"]
    variance = random_walk["variance"]
    market_demand_split = market["demand_split"]

    sc_all = cfg['supply_chain']
    agents_per_level = sc_all['agents_per_level']
    sc_levels = sc_all['sc_levels']

    sc_agent_list = []
    sc_adjacency = []
    sc_lead_time = []

    if data_source is None:
        # Init Simulation
        simulation = Simulation(T=T, sim_runs=sim_runs, sim_time=sim_time, conv_time=conv_time, retraining_time=training_time, 
                                testing_time=testing_time, training_type=training_type, train_size=train_size, val_size=val_size)
        # Init Market
        market = MarketArtifical(sim_time=T, primary_demand=primary_demand, trend_mag=trend_magnitude,
                        seasonality_mag=seansonality_magnitude, seasonality_freq=seasonality_frequncy,
                        random_walk_mean=mean, random_walk_var=variance, retailer_num=agents_per_level[0], market_demand_split=market_demand_split)
    else:
        #load data from csv
        df = pd.read_excel(data_source) 
        data = df.iloc[0].values
        data = np.array([int(x.replace(',', '')) for x in data])/1000

        # set simulations paramters
        if len(data) < T:
            raise ValueError("Simulation time is too long for the given Dataset")
        if len(data)< training_time:
            raise ValueError("TrainingTime is is too long for the given Dataset")

        simulation = Simulation(T=T, sim_runs=sim_runs, sim_time=sim_time, conv_time=conv_time, retraining_time=training_time, 
                                testing_time=testing_time, training_type=training_type,  train_size=train_size, val_size=val_size)
        
        # create market
        marekt = MarketDataSource(data, data, retailer_num=agents_per_level[0], market_demand_split=market_demand_split)


    # Init Agentes
    for i, level in enumerate(sc_levels):
        num_agents = agents_per_level[i]
        sc = sc_levels[level]
        adjaceny_matrix = np.array(sc["adjaceny_list"])
        lead_time_matrix = np.array(sc["lead_time"])
        sc_adjacency.append(adjaceny_matrix)
        sc_lead_time.append(lead_time_matrix)

        agent_list = []
        for j in range(num_agents):
            agent = Agent(id=j, sc_level=i, adjacency_matrix=adjaceny_matrix,
                          lead_time_matrix=lead_time_matrix, training_time=training_time, cfg=sc, cfg_all=sc_all,
                          sequence_length=sequence_length, eopchs=epochs, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)
            agent_list.append(agent)

        sc_agent_list.append(agent_list)
    # Init Supply Chain
    supply_chain = Supply_Chain(sc_adjacency, sc_lead_time)

    return simulation, market, supply_chain, sc_agent_list, cfg

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]

        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def reset_simulaltion_from_dict(cfg: dict) -> list[Simulation, Market, Supply_Chain, list, dict]:
    """Initate all objects needed for simulations

    Args:
        path (Path):path to config file

    Returns:
        list[Simulation, Market, Supply_Chain, list, dict]: return the initialized objects Simulatio,
        Market and Supply Chain, a list of all agents ordered per supply chain level and the rest of the config file
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
    market = cfg['market']
    data_source = cfg['market']['data_scource']
    primary_demand = market["primary_demand"]
    trend_magnitude = market["trend_magnitude"]
    seansonality_magnitude = market["seasonality_magnitude"]
    seasonality_frequncy = market["seasonality_frequncy"]
    random_walk = market["random_walk"]
    mean = random_walk["mean"]
    variance = random_walk["variance"]
    market_demand_split = market["demand_split"]

    sc_all = cfg['supply_chain']
    agents_per_level = sc_all['agents_per_level']
    sc_levels = sc_all['sc_levels']

    sc_agent_list = []
    sc_adjacency = []
    sc_lead_time = []

    if data_source is None:
        # Init Simulation
        simulation = Simulation(T=T, sim_runs=sim_runs, sim_time=sim_time, conv_time=conv_time, retraining_time=training_time, 
                                testing_time=testing_time, training_type=training_type, train_size=train_size, val_size=val_size)
        # Init Market
        market = MarketArtifical(sim_time=T, primary_demand=primary_demand, trend_mag=trend_magnitude,
                        seasonality_mag=seansonality_magnitude, seasonality_freq=seasonality_frequncy,
                        random_walk_mean=mean, random_walk_var=variance, retailer_num=agents_per_level[0], market_demand_split=market_demand_split)
    else:
        #load data from csv
        df = pd.read_excel(data_source) 
        data = df.iloc[0].values
        data = np.array([int(x.replace(',', '')) for x in data])/1000

        # set simulations paramters
        if len(data) < T:
            raise ValueError("Simulation time is too long for the given Dataset")
        if len(data)< training_time:
            raise ValueError("TrainingTime is is too long for the given Dataset")

        simulation = Simulation(T=T, sim_runs=sim_runs, sim_time=sim_time, conv_time=conv_time, retraining_time=training_time, 
                                testing_time=testing_time, training_type=training_type,  train_size=train_size, val_size=val_size)
        
        # create market
        agents_per_level[0]
        market = MarketDataSource(data, retailer_num=agents_per_level[0], market_demand_split=market_demand_split)
    # Init Agentes
    for i, level in enumerate(sc_levels):
        num_agents = agents_per_level[i]
        sc = sc_levels[level]
        adjaceny_matrix = np.array(sc["adjaceny_list"])
        lead_time_matrix = np.array(sc["lead_time"])
        sc_adjacency.append(adjaceny_matrix)
        sc_lead_time.append(lead_time_matrix)

        agent_list = []
        for j in range(num_agents):
            agent = Agent(id=j, sc_level=i, adjacency_matrix=adjaceny_matrix,
                          lead_time_matrix=lead_time_matrix, training_time=training_time, cfg=sc, cfg_all=sc_all,
                          sequence_length=sequence_length, eopchs=epochs, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)
            agent_list.append(agent)

        sc_agent_list.append(agent_list)
    # Init Supply Chain
    supply_chain = Supply_Chain(sc_adjacency, sc_lead_time)

    return simulation, market, supply_chain, sc_agent_list


def select_gpu():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    if torch.cuda.is_available():
        device = select_least_used_gpu()

    return device

def get_gpu_usage():
    # Run nvidia-smi command to get GPU usage
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    # Decode result and split by newlines to get usage for each GPU
    usage = result.stdout.decode('utf-8').split('\n')
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
    print(f"Selected GPU {least_used_gpu} with usage {usage[least_used_gpu]}%")

    return device