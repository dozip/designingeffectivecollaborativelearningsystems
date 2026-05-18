import logging

import numpy as np

logger = logging.getLogger('logger')


def run_simulation_phase(t, sim_time, simulation, market, supply_chain,
                         sc_agent_list, backend):
    """Run the simulation from time t to sim_time.

    For each timestep:
      - get the market demand for this timestep
      - for each supply-chain level:
          - if this level is the backend's collaborative_level, call
            backend.collaborative_predict and use act_multichannel
          - otherwise call agent.act on each agent
          - propagate shipments to the previous level
      - increment t
    """
    logger.info("Start Simulation with the following settings: ")
    logger.info("Simulation Type: " + str(simulation.training_type))
    logger.info("Simulation Time: " + str(sim_time - t))

    while t < sim_time:

        demand_t = np.array([market.split_demand_on_time(t)])
        # go over every level
        for i in range(len(sc_agent_list)):
            # go over every agent
            demand_per_agent = []
            shipment_per_agent = []

            if backend.collaborative_level == i:
                predictions = backend.collaborative_predict(
                    sc_agent_list[i], demand_t, sc_agent_list, t)
                for j, agent in enumerate(sc_agent_list[i]):
                    demand, shipment = agent.act_multichannel(demand_t[:, j], 0, predictions[j])
                    demand_per_agent.append(demand)
                    shipment_per_agent.append(shipment)
            else:
                for j, agent in enumerate(sc_agent_list[i]):
                    demand, shipment = agent.act(demand_t[:, j], 0)
                    demand_per_agent.append(demand)
                    shipment_per_agent.append(shipment)

            demand_t = np.array(demand_per_agent)
            shipment_t = np.array(shipment_per_agent)
            shipment_sum_per_level = np.sum(shipment_t, axis=0)

            if i == 0:
                pass
            else:
                for k, agent in enumerate(sc_agent_list[i - 1]):
                    agent.set_shipment(shipment_sum_per_level[k])

            # set shipment for last level -> demand is always satisfied
            if i == (len(sc_agent_list) - 1):
                for k, agent in enumerate(sc_agent_list[-1]):
                    agent.set_shipment(demand_t[k][0])

        t += 1
