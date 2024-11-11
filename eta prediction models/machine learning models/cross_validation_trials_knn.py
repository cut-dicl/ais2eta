# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 09:45:35 2021

@author: Brian
"""
from pathlib import Path
import get_and_pre_process_dataset
import ML_models
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate, cross_val_predict
from matplotlib import rcParams
from sklearn.feature_selection import RFECV


#import sys
#import os
#import site
#sys.path.append(os.path.join(os.path.dirname(site.getsitepackages()[0]), "site-packages"))

#from sklearnex import patch_sklearn
#patch_sklearn()


rcParams.update({"figure.autolayout": True})
#pd.set_option("max_columns", None)
pd.options.display.max_rows = 5
curr_dir = Path(__file__).parent.absolute()
output_cnfg_dir = rf"{curr_dir}/config files/output config files"


def get_name_of_metrics_to_evaluate_the_mdl():
    metric_names = (pd.read_csv(
        rf"{output_cnfg_dir}/metrics to evaluate.csv",
        usecols=["name in cross_validate function"],
    ).iloc[:, 0].tolist())
    return metric_names


def get_ypred_yactual_grph_cnfg(mdl_nm, plot_only_when_agnt_eta_invld):
    ypred_yactual_grph_cnfg = (pd.read_csv(
        rf"{output_cnfg_dir}/predicted_vs_actual_graph_config.csv",
        usecols=["Parameter Value"],
    ).iloc[:, 0].tolist())
    prefix = f"{mdl_nm}_mdl_for_entire_route_"
    if plot_only_when_agnt_eta_invld == True:
        prefix = f"{mdl_nm}_mdl_only_when_agnt_eta_invld_"
    ypred_yactual_grph_cnfg[0] = prefix + ypred_yactual_grph_cnfg[0]
    return ypred_yactual_grph_cnfg


def validate_plot_type(plot_type):
    valid_plot_types = {"scatter", "barh"}
    if plot_type not in valid_plot_types:
        raise TypeError("Only these plot types are allowed: ",
                        valid_plot_types)


def generate_and_save_plot(plot_name, plot_num, plot_type, title, xlabel,
                           ylabel, x_axis_values, y_axis_values):
    validate_plot_type(plot_type)
    myplot = plt.figure(int(plot_num))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    if plot_type == "scatter":
        plt.scatter(x_axis_values,
                    y_axis_values,
                    color="red",
                    s=0.2,
                    label="Models prediction")
        plt.plot(y_axis_values.to_numpy()[:, None],
                 y_axis_values.to_numpy()[:, None],
                 linewidth=3,
                 label="Ideal prediction")
    if plot_type == "barh":
        plt.barh(x_axis_values, y_axis_values)
    plt.legend()
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    myplot.savefig(plot_name + ".png", dpi=1000)
    plt.cla()


def map_feature_with_importance_or_rank(regressor_feature_importances,
                                        x_train):
    importances = list(regressor_feature_importances)
    feature_importances = [
        (feature, importance)
        for feature, importance in zip(x_train.columns, importances)
    ]  # zip cols, with importance
    feature_importances = sorted(feature_importances,
                                 key=lambda i: i[1],
                                 reverse=True)
    feature_importances_df = pd.DataFrame(feature_importances,
                                          columns=["Feature", "Importance"])
    return feature_importances_df


def get_feature_importances(mach_learn_mdl, x, y, cv):
    selector = RFECV(mach_learn_mdl, step=1, cv=cv)
    #x, y = get_and_pre_process_dataset.load_and_prepare_dataset(get_invalid_eta_as_null = False,eta_in_hours=True)
    selector = selector.fit(x, y)
    feature_support_status = map_feature_with_importance_or_rank(
        selector.support_, x)
    feature_rank = map_feature_with_importance_or_rank(selector.ranking_, x)
    feature_importance = map_feature_with_importance_or_rank(
        selector.estimator_.feature_importances_,
        x.iloc[:, selector.get_support(indices=True)],
    )
    return feature_support_status, feature_rank, feature_importance


def calc_and_store_feature_importances(mdl_nm, mach_learn_mdl, x, y, cv):
    feature_imp_sts, feature_rank, feature_imp_perc = get_feature_importances(
        mach_learn_mdl, x, y, cv)
    f_name_prefix = rf"{mdl_nm}mdl_{cv}fold_cros_val_feature_"
    feature_imp_sts.to_csv(rf"{f_name_prefix}imp_sts.csv", index=False)
    feature_rank.to_csv(rf"{f_name_prefix}rank.csv", index=False)
    feature_imp_perc.to_csv(rf"{f_name_prefix}imp_perc.csv", index=False)
    print("features_completed")


def store_cross_valiadation_rslts(mdl_nm, cv, scores):
    f_name_prefix = rf"{mdl_nm}mdl_{cv}fold_cros_val_"
    f_name_suffix = "_eval_mtrcs.csv"
    eval_mtrcs_scores_df = pd.DataFrame(scores)
    eval_mtrcs_avg_scores_df = pd.DataFrame(
        eval_mtrcs_scores_df.mean().reset_index()).iloc[2:, :]
    eval_mtrcs_avg_scores_df.columns = [
        "evaluation metric",
        "cross validation average score",
    ]
    eval_mtrcs_scores_df.to_csv(rf"{f_name_prefix}scores_for{f_name_suffix}",
                                index=False)
    eval_mtrcs_avg_scores_df.to_csv(
        rf"{f_name_prefix}avg_scores_of{f_name_suffix}", index=False)
    print(eval_mtrcs_scores_df)
    print(eval_mtrcs_avg_scores_df)


def main():
    mdl_nm, cv = "knn", 10
    x, y = get_and_pre_process_dataset.load_and_prepare_dataset(
        get_invalid_eta_as_null=False, eta_in_hours=1)
    print ("Getting Dataset...Done!")
    mach_learn_mdl = ML_models.get_tuned_model(mdl_nm)
    print ("Getting Model...Done!")
    #calc_and_store_feature_importances(mdl_nm, mach_learn_mdl, x, y, cv)
    print ("Calulate and store features...Done!")
    y_pred = cross_val_predict(mach_learn_mdl, x, y, cv=cv)
    print ("Y_pred calulation...Done!")
    indx_invld_eta_agnt = x.index[x["agent_eta_in_min"] == -1].tolist()
    generate_and_save_plot(
        *get_ypred_yactual_grph_cnfg(mdl_nm,
                                     plot_only_when_agnt_eta_invld=True),
        y_pred[indx_invld_eta_agnt],
        y[y.index.isin(indx_invld_eta_agnt)],
    )
    generate_and_save_plot(
        *get_ypred_yactual_grph_cnfg(mdl_nm,
                                     plot_only_when_agnt_eta_invld=False),
        y_pred,
        y,
    )
    scores = cross_validate(mach_learn_mdl,
                            x,
                            y,
                            cv=cv,
                            scoring=get_name_of_metrics_to_evaluate_the_mdl())

    store_cross_valiadation_rslts(mdl_nm, cv, scores)


if __name__ == "__main__":
    main()
