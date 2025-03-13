import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


def _is_unique(x):
    """just check if the vec contins only unique elements"""
    return len(set(x)) == len(x)


class OptimalMatcher:
    def __init__(self, ctl: pd.DataFrame, trt: pd.DataFrame):
        assert _is_unique(ctl.index)
        assert _is_unique(trt.index)
        self.ctl = ctl
        self.trt = trt
        self.nn_graph = None
        self.Gflow = None

    def construct_knn_graph(self, n_neighbors):
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn_graph = nn.fit(self.ctl)

    def construct_flow_graph(self, n_max):
        """turns itself into a networkx graph whose MaxFlow is our desired assignemtn"""
        dist, inx = self.nn_graph.kneighbors(self.trt)

        Gflow = construct_bipart_for_flow(
            dist, inx, self.trt.shape[0], self.ctl.shape[0]
        )
        # add sink, srouce
        self.Gflow = add_source_sink(Gflow, n_max)

    def solve_flow(self):
        flow = nx.max_flow_min_cost(self.Gflow, 'source','sink',capacity='capacity')

        flow = nx.max_flow_min_cost(
            self.Gflow, "source", "sink", capacity="capacity", weight="cost"
        )
        # mapping is at the level of the nodenames i.e. `ctl_i`, `trt_j`
        mapping = flow_to_mapping(flow)

        total_cost = sum(
            self.Gflow[trt_node][ctl_node]["cost_original"]
            for trt_node, ctl_node in mapping.items()
        )

        # translate the nodenames to the index in self.trt, self.ctl
        def _nodename_to_index(name):
            return int(name.split("_")[1])

        trt_index = self.trt.index
        ctl_index = self.ctl.index
        mapping_final = {
            trt_index[_nodename_to_index(trt_i)]: ctl_index[_nodename_to_index(ctl_j)]
            for trt_i, ctl_j in mapping.items()
        }

        return mapping_final, total_cost


def construct_bipart_for_flow(dist, inx, n_trt, n_ctl):
    """
    edges go from source -> (TRT -> CTL) -> sink
    NOTE: we DONT add source/sink here
    """
    assert dist.shape[0] == inx.shape[0] == n_trt

    G = nx.DiGraph()
    [G.add_node(f"ctl_{i}", bipartite=0, nodetype="ctl") for i in range(n_ctl)]
    [G.add_node(f"trt_{i}", bipartite=1, nodetype="trt") for i in range(n_trt)]

    edges = []

    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            d = dist[i, j]
            target_ctl = inx[i, j]
            start_node = f"trt_{i}"
            end_node = f"ctl_{target_ctl}"
            assert start_node in G and end_node in G
            edges.append(
                (
                    start_node,
                    end_node,
                    {"cost": int(10000 * d), "cost_original": d, "capacity": np.inf},
                )
            )

    G.add_edges_from(edges)
    # remove unused ctls
    to_removed = [n for n in G.nodes() if G.degree[n] == 0 and n.startswith("ctl")]
    # print(to_removed)
    G.remove_nodes_from(to_removed)

    return G


def add_source_sink(G, n_max, demand=None):
    """
    augmenting the bipartite CTL-TRT graph with source and sink,
    enabling maxflow algoirithms

    optionally set a demand on the source (negative) and sink
    """

    if demand is None:
        G.add_node("source", nodetype="source", bipartite=1)
        G.add_node("sink", nodetype="sink", bipartite=0)
    else:
        G.add_node("source", nodetype="source", bipartite=1, demand=-demand)
        G.add_node("sink", nodetype="sink", bipartite=0, demand=demand)

    ctl_nodes = [n for n in G if n.startswith("ctl")]
    trt_nodes = [n for n in G if n.startswith("trt")]

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

    trt_nodes = [_ for _ in flow.keys() if _.startswith("trt_")]
    for trt_node in trt_nodes:
        # for trt_node in [f'trt_{i}' for i in range(trt.shape[0])]:
        for ctl_node, active_flow in flow[trt_node].items():
            if active_flow > 0:
                assert trt_node not in mapping
                mapping[trt_node] = ctl_node
    return mapping
