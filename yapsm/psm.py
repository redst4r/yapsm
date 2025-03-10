import numpy as np
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm
import pandas as pd
import patsy
from sklearn.neighbors import NearestNeighbors
from yapsm import optimal_match


def _escape_varname(x):
    # return "Q('{}')".format(x)
    return x


class Yapsm(object):
    @staticmethod
    def _scores_to_accuracy(m, X, y):
        preds = [[1.0 if i >= 0.5 else 0.0 for i in m.predict(X)]]
        return (y.to_numpy().T == preds).sum() * 1.0 / len(y)

    def balanced_sample(self, data=None):
        if not data:
            data = self.data
        minor, major = (
            data[data[self.yvar] == self.group_trt],
            data[data[self.yvar] == self.group_ctl],
        )
        return pd.concat([major.sample(len(minor)), minor]).dropna()

    def _get_ctl_treat_split(self, df):
        crl = df[df[self.yvar] == self.group_ctl]
        trt = df[df[self.yvar] == self.group_trt]
        return crl, trt

    def __init__(self, data, yvar, group_ctl, group_trt, formula=None, exclude=[]):
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
                print("Fitting Models on Balanced Samples")
                # sample from majority to create balance dataset
                df = self.balanced_sample()
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
            print(
                "\nAverage Accuracy:",
                "{}%".format(round(np.mean(self.model_accuracy) * 100, 2)),
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

    def match_1nn(self):
        ctl, trt = self._get_ctl_treat_split(self.data)

        ctl_score = ctl[["scores"]]
        trt_score = trt[["scores"]]

        nn = NearestNeighbors(n_neighbors=1)
        nn = nn.fit(ctl_score)
        dist, inx = nn.kneighbors(trt_score)

        mapping_1nn = {trt.index[i]: ctl.index[inx[i, 0]] for i in range(trt.shape[0])}

        total_distance = dist[:, 0].sum()

        nctl = len(set(mapping_1nn.values()))
        ctl_total = len(ctl)
        print(f"#control samples used {nctl}/{ctl_total}")
        print("Total cost", total_distance)
        return mapping_1nn

    def match_optimal(self, knn, n_max):
        ctl, trt = self._get_ctl_treat_split(self.data)
        ctl_score = ctl[["scores"]]
        trt_score = trt[["scores"]]

        om = optimal_match.OptimalMatcher(ctl_score, trt_score)
        om.construct_knn_graph(knn)
        om.construct_flow_graph(n_max=n_max)
        mapping, total_cost = om.solve_flow()

        print("#control samples used", len(set(mapping.values())))
        print("Total cost", total_cost)

        return mapping


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


def generate_toydata(n_ctl, n_trt):
    x_ctl = np.random.multivariate_normal(
        [0, 0], np.array([[1, 0], [0, 1]]), size=n_ctl
    )
    ctl = pd.DataFrame(x_ctl, columns=["x1", "x2"])
    # ctl = pd.DataFrame(np.random.normal(0,1, size=(n_ctl, 2)), columns=['x1','x2'])
    ctl.index = [f"patient_ctl_{i}" for i in range(ctl.shape[0])]
    ctl["group"] = 0  #'ctl'

    x_trt = np.random.multivariate_normal(
        [2, 0], np.array([[1, 0], [0, 1]]), size=n_ctl
    )
    trt = pd.DataFrame(x_trt, columns=["x1", "x2"])
    trt["group"] = 1  # 'trt'
    trt.index = [f"patient_trt_{i}" for i in range(trt.shape[0])]

    return ctl, trt
