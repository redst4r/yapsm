Yet another Propensity Score matching library. The existing libraries just didn't do it:
- [psmpy](https://pypi.org/project/psmpy/) tries to do an exhaustive match between all controls and all treatments, which is way too slow (`O(n^2)`)
- [pymatch](https://github.com/benmiroglio/pymatch) has a nice way of estimating propensity scores (repeated downsampling and averaging predictions), but just does random (within a threshold) or minimum matching. Also it can't do "no replacement" (i.e. can use the same sample multiple times)

This package uses kNN-graphs (i.e. for a given treatment sample, we consider only it's "k-best" matches in the controls), cutting down the runtime to `O(nk)`.
As for matching, it uses either:
- **simple 1NN match** (i.e. greedily selects the best control for each treatment)
- **optimal matching** (based on `MinCostMaxFlow`), which also allows to specify how often a control is allowed to be reused. Optimal matching is a bit **time consuming** though!

## Minimal example
using **optimal matching**
```python
# make up some data in 2D
ctl = pd.DataFrame(np.random.normal(0, 2, size=(20, 2)), index=[f"ctl_{i}" for i in range(20)], columns=['x1','x2'])
trt = pd.DataFrame(np.random.normal(1, 1, size=(10, 2)), index=[f"trt_{i}" for i in range(10)], columns=['x1','x2'])
ctl['group'] = 0  # indicating whats treatment and whats control
trt['group'] = 1  # no need to stick to the 0/1 encoding
data = pd.concat([ctl, trt])

from yapsm import psm
p = psm.Yapsm(
    data,
    yvar="group",
    group_ctl=0,  #'ctl',
    group_trt=1,  #'trt'
)
p.fit_scores(balance=True)  # fit the LogReg, using class-balanced data
p.predict_scores()  # predict propensity scores
optimal_matching = p.match_optimal(knn=5, n_max=2, caliper=0.25)  # allowing each control to be used 2x at max
df_matched = p.get_psmatched_dataset(optimal_matching) # a dataframe containing the matched dataset
```

Alternatively, use greedy **1-NN matching**
```python
nn1_matching = p.match_1nn(caliper=0.25)
df_matched = p.get_psmatched_dataset(nn1_matching)
```

## Credits
This package is using large chunks of [pymatch's](https://github.com/benmiroglio/pymatch) logistic regression fitting!
