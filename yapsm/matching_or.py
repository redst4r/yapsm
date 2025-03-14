import numpy as np
import networkx as nx
from ortools.graph.python import min_cost_flow, max_flow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ORMatcher:
    """
    Solving the assignment problem / Graph flow with `ortools`,
    which is oreders of magnitude faster than `networkx`
    """

    def __init__(self, Gflow: nx.DiGraph):
        (
            self.start_nodes,
            self.end_nodes,
            self.capacities,
            self.unit_costs,
            self.nodes_mapping,
        ) = convert_to_or(Gflow)

        # to map from ints to actual node names as used in Gflow
        self.nodes_mapping_inv = {i: name for name, i in self.nodes_mapping.items()}

    def solve_flow(self):
        """
        solves the flow problem with minimum cost and converts the solution
        into an optimal assignment

        as part, we need to determine the MaxFlow (without cost)
        """
        smcf = min_cost_flow.SimpleMinCostFlow()

        # Add arcs, capacities and costs in bulk using numpy.
        all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
            self.start_nodes, self.end_nodes, self.capacities, self.unit_costs
        )

        # Add supply for each nodes.
        # this is done be looking at the maxflow thats possible acorss the network, irrespective of cost
        maxflow = self._solve_maxflow()

        supply = maxflow

        def _supply(x):
            if x == "source":
                return supply
            elif x == "sink":
                return -supply
            else:
                return 0

        # supplies = [_supply(n) for n in Gflow.nodes]
        n_nodes = len(self.nodes_mapping)
        supplies = np.array(
            [_supply(self.nodes_mapping_inv[i]) for i in range(n_nodes)]
        )

        smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

        status = smcf.solve()

        if status != smcf.OPTIMAL:
            logger.error("There was an issue with the min cost flow input.")
            logger.error(f"Status: {status}")
            raise ValueError("infeasible flow")
        min_cost = smcf.optimal_cost()
        
        # this cost is inflated due to our integer encoding
        #logger.info(f"Minimum cost: {min_cost}")

        solution_flows = smcf.flows(all_arcs)

        # get the flow dict
        flow_dict = or_solution_to_dict(
            smcf, solution_flows, all_arcs, self.nodes_mapping_inv
        )
        return flow_dict, min_cost

    def _solve_maxflow(self):
        """
        get the maxflow possible from source to sink (given the edge capacities, neglecting cost)
        """
        smf = max_flow.SimpleMaxFlow()
        smf.add_arcs_with_capacity(self.start_nodes, self.end_nodes, self.capacities)

        source_id = self.nodes_mapping["source"]
        sink_id = self.nodes_mapping["sink"]
        status = smf.solve(source_id, sink_id)

        if status != smf.OPTIMAL:
            logger.error("There was an issue with the max flow input.")
            logger.error(f"Status: {status}")
            raise ValueError(f"Status: {status}")
        maxflow = smf.optimal_flow()
        return maxflow


def or_solution_to_dict(smcf, solution_flows, all_arcs, nodes_mapping_inv):
    """
    turns the ORtools solution_flows into a mapping dict, i.e. which trt got link to which control
    """
    # costs = solution_flows * unit_costs
    flow_dict = {}
    for arc, flow in zip(all_arcs, solution_flows):
        tail_id = smcf.tail(arc)
        head_id = smcf.head(arc)

        tail_name = nodes_mapping_inv[tail_id]
        head_name = nodes_mapping_inv[head_id]
        if tail_name == "source" or head_name == "sink":
            continue
        if flow > 0:
            flow_dict[tail_name] = head_name

    return flow_dict


def convert_to_or(Gflow: nx.DiGraph):
    """turn the nx graph into a ORtools instance
    (or rather: the format needed for ortools, which is a flat list of edges, capacities and cost)

    note that this does not set the supply/demand on the nodes (to keep it more generic)
    """

    start_names, end_names, capacities, unit_costs = zip(
        *[
            (u, v, data["capacity"], data["cost"])
            for u, v, data in Gflow.edges(data=True)
        ]
    )

    # the bipartite edges have infinite capacity, which doesnt work with ORtools
    # jsut set them to the max.capacity that could flow in a single edge (which is n_max!)
    # clip it to the maximum capacity of the rest of the graph; since all source-trt have cap 1, there actually never can be more than a flow of one on the edge!
    # even if the input from the source is more than 1 per edge, theres only one edge from source anyways
    capacities = np.array(capacities)
    max_cap = np.max(capacities[~np.isinf(capacities)])
    #logging.info("max_cap", max_cap)  # TODO: works for now, but if we generalize the source/sink connections to arbitrary capacity, this might not work
    capacities = np.clip(capacities, 0, max_cap).astype(np.int64)

    # need to translate the nodenames into ints
    nodes_mapping = {name: i for i, name in enumerate(Gflow.nodes)}
    start_nodes = np.array([nodes_mapping[name] for name in start_names])
    end_nodes = np.array([nodes_mapping[name] for name in end_names])
    unit_costs = np.array(unit_costs).astype(np.int64)

    return start_nodes, end_nodes, capacities, unit_costs, nodes_mapping
