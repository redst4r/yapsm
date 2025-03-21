import numpy as np
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm
import pandas as pd
import patsy
from sklearn.neighbors import NearestNeighbors
from yapsm import optimal_match
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _escape_varname(x):
    # return "Q('{}')".format(x)
    return x


class Yapsm(object):
    @staticmethod
    def _scores_to_accuracy(m, X, y):
        preds = [[1.0 if i >= 0.5 else 0.0 for i in m.predict(X)]]
        return (y.to_numpy().T == preds).sum() * 1.0 / len(y)

    def _balanced_sample(self, data=None):
        if not data:
            data = self.data
        minor, major = (
            data[data[self.yvar] == self.group_trt],
            data[data[self.yvar] == self.group_ctl],
        )
        return pd.concat([major.sample(len(minor)), minor]).dropna()

    def _get_ctl_treat_split(self, df):
        """
        get two separate dataframes, one for control samples, one for treatment
        """
        crl = df[df[self.yvar] == self.group_ctl]
        trt = df[df[self.yvar] == self.group_trt]
        return crl, trt

    def __init__(self, data, yvar, group_ctl, group_trt, formula=None, exclude=[]):
        """
        :param data: DataFrame, one sample per row
        :param yvar: columnname in `data` that indicates group membership
        :param group_ctl: label for the control group, i.e. `data[yvar] == group_ctl` are the control samples
        :param group_ctl: label for the treatment group, see above
        :param formula: can leave empty TODO
        :param exclude: columns to exclude from the logreg TODO
        """
        self.models = []  # the logreg models
        self.model_accuracy = []

        self.group_ctl = group_ctl
        self.group_trt = group_trt
        self.yvar = yvar
        self.exclude = exclude
        self.data = data
        self.formula = formula
        self.xvars = [
            i for i in self.data.columns if i not in self.exclude and i != yvar
        ]

        # self.xvars_escaped = [ "Q('{}')".format(x) for x in self.xvars]
        # self.yvar_escaped = "Q('{}')".format(self.yvar)
        self.xvars_escaped = [_escape_varname(_) for _ in self.xvars]
        self.yvar_escaped = _escape_varname(self.yvar)

        self.y, self.X = patsy.dmatrices(
            "{} ~ {}".format(self.yvar_escaped, "+".join(self.xvars_escaped)),
            data=self.data,
            return_type="dataframe",
        )
        self.xvars = [i for i in self.data.columns if i not in self.exclude]

    def fit_scores(self, balance=True, nmodels=None):
        """
        taken from `pymatch` package

        Fits logistic regression model(s) used for
        generating propensity scores

        Parameters
        ----------
        balance : bool
            Should balanced datasets be used?
            (n_control == n_test)
        nmodels : int
            How many models should be fit?
            Score becomes the average of the <nmodels> models if nmodels > 1

        Returns
        -------
        None
        """
        # reset models if refitting
        if len(self.models) > 0:
            self.models = []
        if len(self.model_accuracy) > 0:
            self.model_accuracy = []
        if not self.formula:
            # use all columns in the model
            self.xvars_escaped = [_escape_varname(_) for _ in self.xvars]
            self.yvar_escaped = _escape_varname(self.yvar)
            self.formula = "{} ~ {}".format(
                self.yvar_escaped, "+".join(self.xvars_escaped)
            )
        if balance:
            if nmodels is None:
                # fit multiple models based on imbalance severity (rounded up to nearest tenth)
                major, minor = self._get_ctl_treat_split(self.data)

                # minor, major = [self.data[self.data[self.yvar] == i] for i in (self.group_trt,
                #                                                                self.group_ctl)]
                nmodels = int(np.ceil((len(major) / len(minor)) / 10) * 10)
            self.nmodels = nmodels
            i = 0
            errors = 0
            while i < nmodels and errors < 5:
                # uf.progress(i+1, nmodels, prestr="Fitting Models on Balanced Samples")
                # print("Fitting Models on Balanced Samples")
                # sample from majority to create balance dataset
                df = self._balanced_sample()
                ctl, trt = self._get_ctl_treat_split(df)
                df = pd.concat(
                    [
                        drop_static_cols(ctl, yvar=self.yvar),
                        drop_static_cols(trt, yvar=self.yvar),
                    ],
                    sort=True,
                )
                y_samp, X_samp = patsy.dmatrices(
                    self.formula, data=df, return_type="dataframe"
                )
                X_samp.drop(self.yvar, axis=1, errors="ignore", inplace=True)
                glm = GLM(y_samp, X_samp, family=sm.families.Binomial())

                try:
                    res = glm.fit()
                    self.model_accuracy.append(
                        self._scores_to_accuracy(res, X_samp, y_samp)
                    )
                    self.models.append(res)
                    i = i + 1
                except Exception as e:
                    errors = (
                        errors + 1
                    )  # to avoid infinite loop for misspecified matrix
                    print("Error: {}".format(e))
            logger.info(
                "Average Classification Accuracy: {}%".format(
                    round(np.mean(self.model_accuracy) * 100, 2)
                ),
            )
        else:
            # ignore any imbalance and fit one model
            print("Fitting 1 (Unbalanced) Model...")
            glm = GLM(self.y, self.X, family=sm.families.Binomial())
            res = glm.fit()
            self.model_accuracy.append(self._scores_to_accuracy(res, self.X, self.y))
            self.models.append(res)
            print("\nAccuracy", round(np.mean(self.model_accuracy[0]) * 100, 2))

    def predict_scores(self):
        """
        Predict Propensity scores for each observation.
        Adds a "scores" columns to self.data

        Returns
        -------
        None
        """
        scores = np.zeros(len(self.X))
        for i in range(self.nmodels):
            m = self.models[i]
            scores += m.predict(self.X[m.params.index])
        self.data["scores"] = scores / self.nmodels

    def match_1nn(self, caliper=np.inf):
        """
        do PSM with a 1st nearest neighbor approach, i.e. for each treatment, pick it;s closest control sample
        Note: controls can be selected multiple times

        :param caliper: max allowed distance for a valid pairing; if the 1-nn is fuarther than `caliper`, the sample wont be paired
        """
        ctl, trt = self._get_ctl_treat_split(self.data)

        ctl_score = ctl[["scores"]]
        trt_score = trt[["scores"]]

        nn = NearestNeighbors(n_neighbors=1)
        nn = nn.fit(ctl_score)
        dist, inx = nn.kneighbors(trt_score)

        mapping_1nn = {}
        total_distance = 0
        for i in range(trt.shape[0]):
            # clipping of matches that are too far apart
            if dist[i, 0] < caliper:
                mapping_1nn[trt.index[i]] = ctl.index[inx[i, 0]]
                total_distance += dist[i, 0]

        N_mapped_trt = len(set(mapping_1nn.keys()))
        N_mapped_ctl = len(set(mapping_1nn.values()))
        logger.info(f"Mapped {N_mapped_trt} TRT  to {N_mapped_ctl} CTL")
        logger.info(f"Total cost {total_distance:.3f}")
        return mapping_1nn

    def match_optimal(self, knn, n_max, caliper=np.inf):
        """PSM with (approximate) optimal matching.
        Approximate: we only consider the knn-Neighbours of a treatment instead of all controls.
        Optimal: restricted to the knn of all the treatments, the solution is optimal (rather than greedy)

        :param knn: number of nearset neighbours to consider per treatment, usually 100 is a good start
        :param n_max: max number of times a control sample can be paired. Set `n_max=1` for 1:1 pairng (i.e. no replacement)
        :param caliper: Any control more distant than `caliper` wont be consider for at treatment
        """
        ctl, trt = self._get_ctl_treat_split(self.data)
        ctl_score = ctl[["scores"]]
        trt_score = trt[["scores"]]

        om = optimal_match.OptimalMatcher(ctl_score, trt_score)
        om.construct_knn_graph(knn)
        om.construct_flow_graph(n_max=n_max, caliper=caliper)

        # mapping, total_cost = om.solve_flow_networkx()  # slow version!
        mapping, total_cost = om.solve_flow_ortools()

        N_mapped_trt = len(set(mapping.keys()))
        N_mapped_ctl = len(set(mapping.values()))
        logger.info(f"Mapped {N_mapped_trt} TRT  to {N_mapped_ctl} CTL")
        logger.info(f"Total cost {total_cost:.3f}")

        return mapping

    def get_psmatched_dataset(self, mapping):
        """simply return the dataframe with the matched samples

        :param mapping: the ctl/trt mapping obtained by `match_optimal` or `match_1nn()`
        """
        indices = set(mapping.keys()) | set(mapping.values())
        return self.data.loc[list(indices)]


def drop_static_cols(df, yvar, cols=None):
    if not cols:
        cols = list(df.columns)
    # will be static for both groups
    cols.pop(cols.index(yvar))
    for col in df[cols]:
        n_unique = len(np.unique(df[col]))
        if n_unique == 1:
            df.drop(col, axis=1, inplace=True)
            # sys.stdout.write('\rStatic column dropped: {}'.format(col))
    return df


def match_1nn_smarter(ctl_score, trt_score, iterations=10):
    """
    with 1NN what happens often is that many trt are matched to a single ctl.
    lets avoid that by (Greedily) reassigning those to the next-best match

    TODO
    """
    nn = NearestNeighbors(n_neighbors=iterations)
    nn = nn.fit(ctl_score)
    dist, inx = nn.kneighbors(trt_score)

    used_ctls = set()
    already_matched_trts = set()
    mapping = {}
    for i in iterations:
        for trt in range(inx.shape[0]):
            if trt in already_matched_trts:
                continue
            ctl = inx[trt, i]  # it's ith best match
            if ctl not in used_ctls:
                mapping[trt] = ctl
                already_matched_trts.add(trt)
                used_ctls.add(ctl)
