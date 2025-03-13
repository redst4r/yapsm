import pandas as pd
import numpy as np
import plotnine as pn
from scipy.special import expit as sigmoid


def summary_imbalance(df, treatment_col):
    """group by treament; for each variable, difference of means between groups, ratio of variances in the groups"""
    avg_group_diffs = df.groupby(treatment_col).mean().T
    avg_group_diffs["mean_diff"] = avg_group_diffs[1] - avg_group_diffs[0]

    avg_group_diffs["std"] = df.std(0)
    avg_group_diffs["std_mean_diff"] = (
        avg_group_diffs["mean_diff"] / avg_group_diffs["std"]
    )

    # STD only in treatment
    avg_group_diffs["std_treat"] = df.query(f"{treatment_col}==1").std(0)
    avg_group_diffs["std_treat_mean_diff"] = (
        avg_group_diffs["mean_diff"] / avg_group_diffs["std_treat"]
    )

    group_var = df.groupby(treatment_col).var().T
    group_var["var_ratio"] = group_var[1] / group_var[0]
    group_var = group_var.drop([0, 1], axis=1)

    return pd.concat([avg_group_diffs, group_var], axis=1)  # .drop('std', axis=1)


def plot_imbalance(df_original, df_matched, treat):
    summary_orig = summary_imbalance(df_original, treat)
    summary_match = summary_imbalance(df_matched, treat)

    melt_orig = summary_orig[["std_mean_diff"]].reset_index()
    melt_orig["source"] = "original"

    melt_matched = summary_match[["std_mean_diff"]].reset_index()
    melt_matched["source"] = "matched"

    df_merged = pd.concat([melt_orig, melt_matched])
    df_merged["abs_mean_diff"] = np.abs(df_merged["std_mean_diff"])
    return (
        pn.ggplot(df_merged)
        + pn.aes(y="index", x="abs_mean_diff", fill="source")
        + pn.geom_point()
    )


def load_lalonde():
    df = pd.read_csv("/home/michi/psmpy/notebooks/lalonde.csv", index_col=0)
    # df['patient'] = [f"patient_{i}" for i in np.arange(0, df.shape[0])]
    df["racewhite"] = df["race"] == "white"
    df["racehispan"] = df["race"] == "hispan"
    df["raceblack"] = df["race"] == "black"
    df = df.drop("race", axis=1)
    return df


def generate_toydata(n_ctl, n_trt):
    # x_ctl = np.random.multivariate_normal(
    #     [0, 0], np.array([[1, 0], [0, 1]]), size=n_ctl
    # )
    x_ctl = np.random.multivariate_normal(
        [-1, 0], np.array([1, 0.5, 0.5, 1]).reshape(2, 2), size=n_ctl
    )

    ctl = pd.DataFrame(x_ctl, columns=["x1", "x2"])
    # ctl = pd.DataFrame(np.random.normal(0,1, size=(n_ctl, 2)), columns=['x1','x2'])
    ctl.index = [f"patient_ctl_{i}" for i in range(ctl.shape[0])]
    ctl["group"] = 0  #'ctl'

    # x_trt = np.random.multivariate_normal(
    #     [2, 0], np.array([[1, 0], [0, 1]]), size=n_ctl
    # )
    x_trt = np.random.multivariate_normal(
        [-1, 2.5], np.array([1, 0.5, 0.5, 1]).reshape(2, 2), size=n_trt
    )
    trt = pd.DataFrame(x_trt, columns=["x1", "x2"])
    trt["group"] = 1  # 'trt'
    trt.index = [f"patient_trt_{i}" for i in range(trt.shape[0])]

    ctl["outcome"] = np.random.binomial(n=1, p=sigmoid(ctl["x2"]))
    trt["outcome"] = np.random.binomial(n=1, p=sigmoid(trt["x2"]))
    return ctl, trt
