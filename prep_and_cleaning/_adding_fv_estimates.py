# TO BE CLEANED -----------------------------

# Modules
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
# ----------------------------------


def add_cheah_2015(df):
    df["fv_c&f_2015"] = 0
    print("|---| Added the Cheah and Fry 2015 FV Estimate |---| ")
    return df


def add_hayes_2018(df
                   , settings
                   , col_hash_rate="hashrate"
                   , col_btc_mined_per_day="inflation"
                   , col_efficiency="efficiency"):
    # hashing power
    hash_rate_in_tera_hash_per_second = df[col_hash_rate]
    hash_rate_in_giga_hash_per_second = hash_rate_in_tera_hash_per_second*1000
    # efficiency
    efficiency_in_joule_per_giga_hash = df[col_efficiency]
    efficiency_in_watt_per_giga_hash_per_second = efficiency_in_joule_per_giga_hash # watt = joule per second
    # energy consumption
    energy_consumption_in_watt=hash_rate_in_giga_hash_per_second * efficiency_in_watt_per_giga_hash_per_second
    energy_consumption_in_kilowatt_hours=energy_consumption_in_watt/1000 * 24
    # expenditure for consumption
    energy_cost_in_usd_per_kilowatt_per_hour = settings["prep_and_cleaning"]["fvalue"]["h_2018"]["energycost_per_kwh"]
    agg_energy_cost_in_usd = energy_consumption_in_kilowatt_hours * energy_cost_in_usd_per_kilowatt_per_hour
    # produced BTC per day
    bitcoin_mined_per_day_in_btc = df[col_btc_mined_per_day]
    # costs per BTC
    cost_per_btc = agg_energy_cost_in_usd / bitcoin_mined_per_day_in_btc

    df["fv_h_2018"] = cost_per_btc

    print("|---| Added Hayes 2018 FV Estimate |---| ")
    return df


def add_fv_caginalp_and_caginalp_2018(col_trend_base
    , df
    , settings
    , precision
    , scaling_variable="from_config"
    , minimal_observation_number="from_config"
    ):

    # Input either from config file or specified in arguments
    if minimal_observation_number == "from_config":
        minimal_observation_number = settings["prep_and_cleaning"]["fvalue"]["c_and_c_2018"]["minobs"]
    elif not type(minimal_observation_number) == int:
        raise TypeError("Input has to be an integer.")

    if scaling_variable == "from_config":
         scaling_variable = settings["prep_and_cleaning"]["fvalue"]["c_and_c_2018"]["scaling"]
    elif not type(scaling_variable) == int:
        raise TypeError("Input has to be an integer.")

    if precision == "from_config":
        precision = settings["prep_and_cleaning"]["fvalue"]["c_and_c_2018"]["precision"]
    elif not type(precision) == int:
        raise TypeError("Input has to be an integer.")

    # 1) Some preparation
    trend_base_variable = df[col_trend_base].copy()

    fv = list()
    for idx in range(len(trend_base_variable)):
        if idx > minimal_observation_number:
            i = idx
            sum_for_period = 0
            while i >= 0:
                value  = trend_base_variable[i]
                weight = np.exp(-(idx-i)/scaling_variable)
                if abs(weight) < 0.1:#precision:
                    break
                sum_for_period += (1 / scaling_variable) * weight * value
                i = i - 1
                #print(i)
            fv.append(sum_for_period)
        else:
            # otherwise trend is NA fo this period
            fv.append(np.nan)

    # Add trend variable to original df and give back with message
    column_name = 'fv_cc2018_c_'+ str(scaling_variable)
    df[column_name] = pd.Series(fv)

    print("|---| Added C&C 2018 FV Estimate (scaling_variable = " + str(scaling_variable) + ") |---| ")
    return df

# def add_simple_trend(col_trend_base
#     , df
#     , settings
#     , window_length="from_config"
#     , scaling_variable="from_config"
#     , minimal_observation_number="from_config"
#     ):
#
#     # Input either from config file or specified in arguments
#     if window_length == "from_config":
#         window_length = settings["prep_and_cleaning"]["trend_longterm_window_length"]
#     elif not type(window_length) == int:
#         raise TypeError("Input has to be an integer.")
#
#     if minimal_observation_number == "from_config":
#         ols_minimal_observation_number = settings["prep_and_cleaning"]["trend_longterm_ols_minimal_observation_number"]
#     elif not type(minimal_observation_number) == int:
#         raise TypeError("Input has to be an integer.")
#
#     # The trend variable uses weighted sums of historic price changes.
#     # This sum is based on the length of the used time window k.
#     # For calculation we adapt Y steps:
#     # 1) Prepare some variables used in the later loop
#     # 2) Create a list of tuples for selecting elements of the price vector
#     # .. that later used to determine *which* historic prices changes are
#     # .. part of the summand
#     # 3) A nested loop going through every period and within every period
#     # .. trough every tuple to select k summands to include for the trend
#     # .. value for the respective period. (Caginalp 2014, p.10)
#
#     # 1) Some preparation
#     trend_base_variable = df[col_trend_base].copy()
#
#     # 2) Selection tuples of indeces for historic price to be part of sum per period
#     tuple_selections_over_time = _make_tuples(window_length=window_length
#                                               , series=trend_base_variable
#                                               , trend_type="simple")
#     # 3) Loop to concatenate weighted sums of price changes per period
#     trend = list()
#     for tuple_selection_for_period in tuple_selections_over_time:
#         cond_sufficient_obs = len(tuple_selection_for_period) >= minimal_observation_number
#         cond_no_zeros = all([False if trend_base_variable[tupl[1]] == 0 else True
#                              for tupl in tuple_selection_for_period])
#         if cond_sufficient_obs or cond_no_zeros:
#             sum_for_period = 0
#             for k, tupl in enumerate(tuple_selection_for_period):
#
#                 updated_value = trend_base_variable[tupl[0]] #"larger" time if it was timestamp
#                 anchor_value  = trend_base_variable[tupl[1]] #"smaller" time if it was timestamp
#                 weight        = np.exp(scaling_variable*(k+1))# "+1" as enumerate starts with 0
#
#                 sum_for_period += weight * (
#                     (updated_value-anchor_value) / anchor_value
#                 )
#
#             trend.append(1/scaling_variable*sum_for_period)
#
#         else:
#             # otherwise trend is NA fo this period
#             trend.append(np.nan)
#
#     # Add trend variable to original df and give back with message
#     column_name = 'trend_' + col_trend_base +'_st_'+ str(window_length)
#     df[column_name] = pd.Series(trend)
#
#     print("|---| Added shorterm trend (window_length = "
#               + str(window_length)
#               + ") < " + col_trend_base + " > |---| ")
#     return df


# def add_ols_trend(col_trend_base
#                         , col_time
#                         , df
#                         , settings
#                         , window_length="from_config"
#                         , periodization_multiplier="from_config"
#                         , ols_minimal_observation_number="from_config"):
#
#     # Input either from config file or specified in arguments
#     if window_length == "from_config":
#         window_length = settings["prep_and_cleaning"]["trend_longterm_window_length"]
#     elif not type(window_length) == int:
#         raise TypeError("Input has to be an integer.")
#
#     if periodization_multiplier == "from_config":
#         periodization_multiplier = settings["prep_and_cleaning"]["trend_longterm_periodization_multiplier"]
#     elif not type(periodization_multiplier) == int:
#         raise TypeError("Input has to be an integer.")
#
#     if ols_minimal_observation_number == "from_config":
#         ols_minimal_observation_number = settings["prep_and_cleaning"]["trend_longterm_ols_minimal_observation_number"]
#     elif not type(ols_minimal_observation_number) == int:
#         raise TypeError("Input has to be an integer.")
#
#     trend_base_variable = df[col_trend_base]
#
#     # Inputs required for long term trend
#     trend_base_variable_changes = trend_base_variable / trend_base_variable.shift(-1) - 1
#     trend_base_variable_changes.index = df[col_time]
#
#     # Loop over time series to apply rolling functionality (OLS here)
#     trend = list()
#     for i in range(len(trend_base_variable_changes)):
#
#         # Determine cut to apply functionality to
#         index_from = i
#         index_to   = window_length + i
#
#         # Extract data for regression
#         regressiondata_endog_var = trend_base_variable_changes.shift(-1)[index_from:index_to]
#         regressiondata_exog_var  = trend_base_variable_changes[index_from:index_to]
#         regressiondata = pd.concat([regressiondata_endog_var,regressiondata_exog_var],
#                                    axis=1)
#         regressiondata.columns = ["trend_base_variable_changes","lagged_trend_base_variable_changes"]
#
#         # Performance of regression on data extract if sufficient observ.
#         cond_nans_data_exog   = regressiondata_exog_var.isna().sum()
#         cond_nans_data_endog  = regressiondata_endog_var.isna().sum()
#         cond_inf_data_exog    = np.isinf(regressiondata_exog_var).any()
#         cond_inf_data_endog   = np.isinf(regressiondata_endog_var).any()
#         cond_to_little_observatios = regressiondata.shape[0] < ols_minimal_observation_number
#
#         if (cond_nans_data_endog or
#             cond_nans_data_exog or
#             cond_inf_data_endog or
#             cond_inf_data_exog or
#             cond_to_little_observatios):
#             slope = np.nan
#         else:
#             slope = sm.ols(formula = "trend_base_variable_changes ~ lagged_trend_base_variable_changes",
#                            data    = regressiondata,
#                            missing = 'drop').fit().params[1]
#
#         slope_periodized = slope * periodization_multiplier
#         trend.append(slope_periodized)
#
#     # Add trend variable to original df and give back with message
#     column_name = 'trend_'+ col_trend_base +'_ols_'+ str(window_length)
#     df[column_name] = pd.Series(trend)
#
#     print("|---| Added long term trend (window_length = "
#           + str(window_length)
#           + ") < " + col_trend_base + " > |---| ")
#
#     return df

