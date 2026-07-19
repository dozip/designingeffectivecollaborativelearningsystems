from .agent import *
from .market import *
from helpers.helper_classes import *
from .supply_chain import *


class Simulation():
    """Container for simulation timing and training configuration.

    For simulations without model training, the class validates only that
    ``sim_time`` is positive. For simulations with model training, it
    currently validates only that ``retraining_time`` is shorter than
    ``sim_time``.

    The class does not perform a complete feasibility or consistency
    validation of all timing and dataset-size parameters.
    """

    def __init__(self, T, sim_runs: int, sim_time: int, conv_time: int, retraining_time: int, testing_time: int, training_type: str,
                 train_size, val_size):
        self.T = T
        self.sim_runs = sim_runs # iterations of entire simulation
        self.sim_time = sim_time # timesteps after convergence
        self.conv_time = conv_time # warum up
        self.retraining_time = retraining_time # time steps considered for training
        self.testing_time = testing_time # time steps for testing
        self.training_type = training_type
        self.val_size = val_size
        self.train_size = train_size

        if self.training_type is None:
            assert self.sim_time > 0
        else:
            self.__check_sim()

    def __check_sim(self):
        

        assert (self.retraining_time < self.sim_time)  # The retraining window must be shorter than the simulation horizon.

    def run():
        raise NotImplementedError
    

    
