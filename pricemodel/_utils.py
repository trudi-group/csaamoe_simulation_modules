import pandas as pd
import itertools
import numpy as np


def resample_to_daily(df
                      , datecol):
    df[datecol] = pd.to_datetime(df.time)
    df = df.resample(on=datecol, rule="D").last()
    df[datecol] = df.index.values
    df.reset_index(drop=True, inplace=True)
    return df


def test_diff(val_test
            , val_true
            , digits=10):
    error = round(val_test, digits) - round(val_true, digits)
    if error == 0:
        test_result = "passed"
    else:
        test_result = "failed"
    return test_result


def make_grid(memory_param_m_from
              , memory_param_m_to
              , memory_param_m_steps
              , memory_param_v_from
              , memory_param_v_to
              , memory_param_v_steps
              , amplitude_param_v_from
              , amplitude_param_v_to
              , amplitude_param_v_steps
              , amplitude_param_m_from
              , amplitude_param_m_to
              , amplitude_param_m_steps
              , iter_start_from
              , iter_start_to
              , iter_start_steps):
    memory_param_m = pd.Series(np.linspace(memory_param_m_from
                                           , memory_param_m_to
                                           , memory_param_m_steps))
    memory_param_v = pd.Series(np.linspace(memory_param_v_from
                                           , memory_param_v_to
                                           , memory_param_v_steps))
    amplitude_param_m = pd.Series(np.linspace(amplitude_param_m_from
                                                  , amplitude_param_m_to
                                                  , amplitude_param_m_steps))
    amplitude_param_v = pd.Series(np.linspace(amplitude_param_v_from
                                                  , amplitude_param_v_to
                                                  , amplitude_param_v_steps))
    iter_start = pd.Series(np.linspace(iter_start_from
                                       , iter_start_to
                                       , iter_start_steps))
    grid = [memory_param_m
        , memory_param_v
        , amplitude_param_m
        , amplitude_param_v
        , iter_start]
    grid_cases = list(itertools.product(*grid))

    return grid_cases


def clean_colname_from_grid_input(case,typ):
    out = "{}_cm_{}_cv_{}_qm_{}_qv_{}".format(
        typ.upper()
        , str(round(case[0],1)).replace(".", "")
        , str(round(case[1],1)).replace(".", "")
        , str(round(case[2],1)).replace(".", "")
        , str(round(case[3],1)).replace(".", ""))
    return out


