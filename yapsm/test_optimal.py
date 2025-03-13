from . import optimal_match
from .utils import generate_toydata
import networkx as nx

def example_graph():
    G = nx.complete_bipartite_graph(4, 4)
    G = nx.DiGraph()
    for i in range(4):
        G.add_node(f'trt_{i}', bipartite=1)
    for i in range(4,8):
        G.add_node(f'ctl_{i}', bipartite=0)
    G.add_edge('trt_0','ctl_4', cost=10, cost_original=10)
    G.add_edge('trt_0','ctl_5', cost=10, cost_original=10)
    G.add_edge('trt_1','ctl_5', cost=1, cost_original=1)
    G.add_edge('trt_2','ctl_5', cost=10, cost_original=10)
    G.add_edge('trt_3','ctl_6', cost=10, cost_original=10)
    G.add_edge('trt_3','ctl_7', cost=2, cost_original=2)
    nx.set_edge_attributes(G, values=10, name='capacity')
    return G

def test_correctness():
    ctl, trt = generate_toydata(4, 4) # note: this will be overridden anyways
    om = optimal_match.OptimalMatcher(ctl, trt)
    om.construct_knn_graph(4)
    om.construct_flow_graph(n_max=2)

    G = example_graph()
    om.Gflow =  optimal_match.add_source_sink(G.copy(), demand=trt.shape[0], n_max=4)

    optimal_map, total_cost = om.solve_flow()
    print(optimal_map)

    assert optimal_map == {'trt_0': 'ctl_4', 'trt_1': 'ctl_5', 'trt_2': 'ctl_5', 'trt_3': 'ctl_7'}
    assert total_cost == 23


def test_the_whole_thing():
    ctl, trt = generate_toydata(10, 9)

    om = optimal_match.OptimalMatcher(ctl, trt)
    om.construct_knn_graph(5)
    om.construct_flow_graph(n_max=5)
    optimal_map = om.solve_flow()

    print(optimal_map)

    # assert 1==0
