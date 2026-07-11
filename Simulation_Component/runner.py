import logging

import numpy as np

logger = logging.getLogger('logger')


def run_simulation_phase(t, sim_time, simulation, market, supply_chain,
                         sc_agent_list, backend):
    """Run the simulation from time t to sim_time.

    For each timestep:
      - get the market demand for this timestep as R*P streams (R retailers,
        P products) in order s = retailer*P + product, reshaped to (R, P)
      - for each supply-chain level:
          - retailers (level 0) receive their per-product demand vector;
            upstream agents receive their per-channel incoming order vector
          - if this level is the backend's collaborative_level, use
            act_multichannel with collaborative predictions
          - propagate shipments back to the previous level, routing each
            product to the retailer that ordered it
      - increment t
    """
    logger.info("Start Simulation with the following settings: ")
    logger.info("Simulation Type: " + str(simulation.training_type))
    logger.info("Simulation Time: " + str(sim_time - t))

    num_products = int(getattr(supply_chain, "num_products", 1))
    product_to_suppliers = getattr(supply_chain, "product_to_suppliers", None)
    num_retailers = len(sc_agent_list[0])
    last_level = len(sc_agent_list) - 1

    while t < sim_time:

        # Market demand: flat R*P stream (order s = retailer*P + product).
        demand_flat = np.asarray(market.split_demand_on_time(t), dtype=float).reshape(-1)
        expected = num_retailers * num_products
        if demand_flat.size != expected:
            raise ValueError(
                f"Market returned {demand_flat.size} demand streams at t={t}, "
                f"expected num_retailers*num_products = {num_retailers}*{num_products} "
                f"= {expected}."
            )
        # Row r = retailer r's per-product demand vector (length P).
        demand_matrix = demand_flat.reshape(num_retailers, num_products)

        # Orders emitted by the previous level; column j is agent j's incoming
        # per-channel demand at the current level.
        orders_prev = None

        for i in range(len(sc_agent_list)):
            level_agents = sc_agent_list[i]

            # Per-agent demand input for this level.
            if i == 0:
                agent_inputs = [demand_matrix[j] for j in range(len(level_agents))]
                collaborative_demand = demand_matrix  # (only used if level 0 collaborates)
            else:
                agent_inputs = [orders_prev[:, j] for j in range(len(level_agents))]
                collaborative_demand = orders_prev

            demand_per_agent = []
            shipment_per_agent = []

            if backend.collaborative_level == i:
                predictions = backend.collaborative_predict(
                    level_agents, collaborative_demand, sc_agent_list, t)
                for j, agent in enumerate(level_agents):
                    demand, shipment = agent.act_multichannel(
                        agent_inputs[j], 0, predictions[j])
                    demand_per_agent.append(demand)
                    shipment_per_agent.append(shipment)
            else:
                for j, agent in enumerate(level_agents):
                    demand, shipment = agent.act(agent_inputs[j], 0)
                    demand_per_agent.append(demand)
                    shipment_per_agent.append(shipment)

            orders_prev = np.array(demand_per_agent)
            shipment_t = np.array(shipment_per_agent)

            # Propagate shipments back to the previous (downstream) level.
            if i > 0:
                _distribute_shipments(
                    receiving_agents=sc_agent_list[i - 1],
                    shipment_t=shipment_t,
                    receiving_level=i - 1,
                    num_products=num_products,
                    product_to_suppliers=product_to_suppliers,
                )

            # Last level's own demand is always satisfied: it receives back what
            # it ordered from the (implicit) upstream source.
            if i == last_level:
                for k, agent in enumerate(level_agents):
                    agent.set_shipment(orders_prev[k][0])

        t += 1


def _distribute_shipments(receiving_agents, shipment_t, receiving_level,
                          num_products, product_to_suppliers):
    """Deliver shipments from a level to its downstream (receiving) level.

    shipment_t[m, k] is sending-agent m's shipment to receiving-agent k.

    When the receiving agents are multi-product retailers, each product p is
    delivered from the manufacturer(s) that produce it
    (product_to_suppliers[p]), so retailer k's product-p inventory receives
    exactly Σ_{m in product_to_suppliers[p]} shipment_t[m, k]. Single-product
    receivers get the total across all senders, matching the legacy behaviour.
    """
    is_multi_product_retailers = (
        receiving_level == 0 and num_products > 1 and product_to_suppliers is not None
    )

    if not is_multi_product_retailers:
        # Legacy path: sum over senders -> one scalar per receiver.
        shipment_sum = np.sum(shipment_t, axis=0)
        for k, agent in enumerate(receiving_agents):
            agent.set_shipment(shipment_sum[k])
        return

    for k, agent in enumerate(receiving_agents):
        per_product = np.zeros(num_products, dtype=float)
        for p in range(num_products):
            suppliers = product_to_suppliers[p]
            per_product[p] = float(np.sum([shipment_t[m, k] for m in suppliers]))
        agent.set_shipment(per_product)
