import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_unique(x):
    """just check if the vec contins only unique elements"""
    return len(set(x)) == len(x)


class OptimalMatcher:
    """Optimal matching via networkx MaxFlow (based on NetworkSimplex)

    Transforms the matching problem into a Flow problem on graphs
    """

    def __init__(self, ctl: pd.DataFrame, trt: pd.DataFrame):
        """
        :param ctl: Dataframe with control sample; index needs to be unique
        :param trt: Dataframe with treatment sample; index needs to be unique
        """
        assert _is_unique(ctl.index)
        assert _is_unique(trt.index)
        assert len(set(ctl.index) & set(trt.index)) == 0, "ctl and trt ids cant overl"

        self.ctl = ctl
        self.trt = trt
        self.nn_graph = None
        self.Gflow = None

    def construct_knn_graph(self, n_neighbors):
        """build a knn graph of the controls"""
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn_graph = nn.fit(self.ctl)

    def construct_flow_graph(self, n_max, caliper=np.inf):
        """turns itself into a networkx graph whose MaxFlow is our desired assignemtn

        :param n_max: max number of times the same control can be assignet to any treatment (n_max=1 for 1:1 matching)
        :param caliper: ctl/trt pairs with distance > caliper wont be considered; essentially trims edges in the flow graph
        """
        logger.info(f"kNN query (k={self.nn_graph.n_neighbors})")
        dist, inx = self.nn_graph.kneighbors(self.trt)

        logger.info("constructing Flow graph")
        Gflow = construct_bipart_for_flow(
            dist, inx, list(self.trt.index), list(self.ctl.index)
        )
        # add sink, srouce
        self.Gflow = add_source_sink(Gflow, n_max)
        apply_caliper(Gflow, caliper=caliper)

    def solve_flow(self):
        """Solving the network flow problem, thereby returning the optimal assignemnt of ctl/trrt"""
        logger.info("solving flow")

        flow = nx.max_flow_min_cost(
            self.Gflow, "source", "sink", capacity="capacity", weight="cost"
        )
        # mapping is at the level of the nodenames i.e. `ctl_i`, `trt_j`
        mapping = flow_to_mapping(flow)

        total_cost = self.get_cost(mapping)

        return mapping, total_cost

    def get_cost(self, mapping):
        return sum(
            self.Gflow[trt_node][ctl_node]["cost_original"]
            for trt_node, ctl_node in mapping.items()
        )


def construct_bipart_for_flow(dist, inx, trt_names, ctl_names):
    """
    trt_names and ctl names must match the order in dist, and inx

    edges go from source -> (TRT -> CTL) -> sink
    NOTE: we DONT add source/sink here
    """
    assert dist.shape[0] == inx.shape[0] == len(trt_names)
    assert _is_unique(trt_names)
    assert _is_unique(ctl_names)
    assert len(set(trt_names) & set(ctl_names)) == 0, "ctl and trt ids cant overl"

    G = nx.DiGraph()
    [G.add_node(n, bipartite=0, nodetype="ctl") for n in ctl_names]
    [G.add_node(n, bipartite=1, nodetype="trt") for n in trt_names]

    edges = []

    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            d = dist[i, j]
            target_ctl = inx[i, j]
            start_node = trt_names[i]
            end_node = ctl_names[target_ctl]
            assert start_node in G and end_node in G
            edges.append(
                (
                    start_node,
                    end_node,
                    # maxflow can only deal with integer costs, so lets just make them into big ints via rounding
                    {"cost": int(10000 * d), "cost_original": d, "capacity": np.inf},
                )
            )

    G.add_edges_from(edges)
    # remove unused ctls
    to_removed = [
        n
        for n, data in G.nodes(data=True)
        if G.degree[n] == 0 and data["bipartite"] == 0
    ]
    G.remove_nodes_from(to_removed)

    return G


def add_source_sink(G, n_max, demand=None):
    """
    augmenting the bipartite CTL-TRT graph with source and sink,
    enabling maxflow algoirithms

    optionally set a demand on the source (negative) and sink
    """

    ctl_nodes = [n for n, data in G.nodes(data=True) if data["bipartite"] == 0]
    trt_nodes = [n for n, data in G.nodes(data=True) if data["bipartite"] == 1]

    if demand is None:
        G.add_node("source", nodetype="source", bipartite=1)
        G.add_node("sink", nodetype="sink", bipartite=0)
    else:
        G.add_node("source", nodetype="source", bipartite=1, demand=-demand)
        G.add_node("sink", nodetype="sink", bipartite=0, demand=demand)

    # ctl_nodes = [n for n, data in G.nodes(data=True) if data['bipartite']==0]
    # trt_nodes = [n for n, data in G.nodes(data=True) if  data['bipartite']==1]

    edges = []
    for trt in trt_nodes:
        edges.append(("source", trt, {"capacity": 1, "cost": 0, "cost_original": 0}))

    for ctl in ctl_nodes:
        if ctl in G:  # since we removed empty ctls
            edges.append(
                (ctl, "sink", {"capacity": n_max, "cost": 0, "cost_original": 0})
            )

    G.add_edges_from(edges)
    return G


def apply_caliper(Gflow, caliper: float):
    """
    warning: modifies the original
    """
    edges_to_remove = []
    for u, v, data in Gflow.edges(data=True):
        if data["cost_original"] > caliper:
            edges_to_remove.append((u, v))
    Gflow.remove_edges_from(edges_to_remove)


# def do_optimal_match(ctl, trt):

#     nn = NearestNeighbors(n_neighbors=50)
#     nn = nn.fit(ctl)
#     dist, inx = nn.kneighbors(trt)

#     Gflow = construct_bipart_for_flow(dist, inx, n_max=6, n_trt=trt.shape[0], n_ctl=ctl.shape[0])
#     flow = nx.min_cost_flow(Gflow, weight='cost', capacity='capacity')
#     mapping = flow_to_mapping(flow)

#     total_cost = 0
#     for trt_node, ctl_node in mapping.items():
#         total_cost += Gflow[trt_node][ctl_node]['cost_original']

#     return mapping , total_cost


def flow_to_mapping(flow):
    """
    turn the output of `nx.min_cost_flow` (which edges are used etc)
    into the desired optimal mapping [trt]->[ctl]
    """
    """
    keys: treatment nodes
    values: ctl nodes
    """
    mapping = {}

    for node, target_dict in flow.items():
        # for each start node, target_dict has the flow of start - key
        if node == "source":
            continue

        for target, active_flow in target_dict.items():
            if target != "sink" and active_flow > 0:
                assert node not in mapping
                mapping[node] = target

    return mapping
