from . import optimal_match
from .psm import generate_toydata


def test_the_whole_thing():
    ctl, trt = generate_toydata(10, 9)

    om = optimal_match.OptimalMatcher(ctl, trt)
    om.construct_knn_graph(5)
    om.construct_flow_graph(n_max=5)
    optimal_map = om.solve_flow()

    print(optimal_map)

    # assert 1==0
