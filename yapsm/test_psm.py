import pandas as pd
from . import psm
from .psm import generate_toydata


def test_the_whole_thing():
    ctl, trt = generate_toydata(100, 90)

    data = pd.concat([ctl, trt])

    p = psm.Yapsm(
        data,
        yvar="group",
        group_ctl=0,  #'ctl',
        group_trt=1,  #'trt'
    )
    # p.fit_scores(balance=False)

    p.fit_scores(balance=True)

    p.predict_scores()

    p.match_1nn()
    p.match_optimal(knn=20, n_max=5)

    # assert 1==0
