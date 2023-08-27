from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import shutil
import pathlib

from . _components import make_zeta, get_k_from_past, m_hodl_from_k
from . _utils import test_diff, resample_to_daily
from . _wrappers import fv_wrapped, theoretical_price_wrapped


def testing_logistics(dir_test_data
            , dir_logging
            , reset_logs
            , fname_test_data
            , given_prices_calc_fv=True
            , given_fv_calc_prices=True
            , given_fv_and_prices_calc_components=True):
    # ----- Setup Logging ----
    # .. Thanks to https://docs.python.org/3/howto/logging-cookbook.html
    # Logging path handling
    path = pathlib.Path(dir_logging)
    if reset_logs and path.exists():
        shutil.rmtree(dir_logging)
    path.mkdir(parents=True, exist_ok=True)
    # create logger with 'spam_application'
    logger = logging.getLogger('test_results_logistics')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(dir_logging+"test_results_logistics.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # ------ 1. Load time, prices, v_circ, onchain_vol_usd, m_total, and parameters
    inpt = resample_to_daily(pd.read_csv(dir_test_data + fname_test_data), "time")
    # -- 1.1. Get true data
    true_data = inpt.loc[:, ["time"
                                   , "prices"
                                   , "fv"
                                   , "v_circ"
                                   , "onchain_vol_usd"
                                   , "m_total"
                                   , "m_hodl"
                                   , "k"
                                   , "zeta_m"
                                   , "zeta_v"
                                ]]
    # -- dataframe missing data that will be calculated with that module (fv, zetas, k, m_hodl)
    # ... dta will be replaced by NAs before the tests
    dta_to_be_tested = true_data.copy()
    # -- Prepare testing dataset for tests
    # Set variables for induction starts for both directions:
    # ... from FV to price
    # ... from price to FV
    # -- Let "get_data" calculate the data to be tested against true data
    # For constructing test data from t=0 to t=... , the first estimate is flawed, as the first
    # ... constructed price-FV-pair is just arbitrary. Making the price dynamic, would cost starting with t>0
    # ...  in the test data. To understand the complete logic, it made sense to start with t=0
    # ...  and keep the first fv-estimate wrong.
    test_results=list()
    if given_prices_calc_fv:
        # ... erase data to be tested from input
        dta_to_be_tested["fv"] = pd.Series(np.nan, index=range(0, len(dta_to_be_tested)))
        # ... recompute
        test_data = get_fv(input_dta=dta_to_be_tested
                                , memory_param_m=inpt.loc[0, "c_m"]
                                , memory_param_v=inpt.loc[0, "c_v"]
                                , amplitude_param_m=inpt.loc[0, "q_m"]
                                , amplitude_param_v=inpt.loc[0, "q_v"]
                                , idx_start=4
                                , induction_start=inpt.loc[0, "induction_start_fv"]
                                , induction_correction=inpt.loc[0, "induction_correction_fv"]
                                , reset_logs=reset_logs
                                , dir_logging=dir_logging
                                , precision=float(0)
                                , verbose=False
                        )
        # ... compare
        test = test_data.reindex(sorted(test_data.columns), axis=1).round(5)
        true = true_data.reindex(sorted(true_data.columns), axis=1).round(5)
        test_results.append(test.equals(true))

    elif given_fv_calc_prices:
        # ... erase data to be tested from input
        dta_to_be_tested["prices"] = pd.Series(np.nan, index=range(0, len(dta_to_be_tested)))
        # ... recompute
        test_data = get_theoretic_prices(input_dta=dta_to_be_tested
                                , memory_param_m=inpt.loc[0, "c_m"]
                                , memory_param_v=inpt.loc[0, "c_v"]
                                , amplitude_param_m=inpt.loc[0, "q_m"]
                                , amplitude_param_v=inpt.loc[0, "q_v"]
                                , idx_start= 2
                                , induction_start_prices_1=inpt.loc[0, "induction_start_prices_1"]
                                , induction_start_prices_2=inpt.loc[0, "induction_start_prices_2"]
                                , reset_logs=reset_logs
                                , dir_logging=dir_logging
                                , precision=float(0)
                                , verbose=False
                                , truncate_zH_until_period=False
                         )
        # ... compare
        test = test_data.reindex(sorted(test_data.columns), axis=1).round(5)
        true = true_data.reindex(sorted(true_data.columns), axis=1).round(5)
        test_results.append(test.equals(true))
    elif given_fv_and_prices_calc_components:
        # ... erase data to be tested from input
        dta_to_be_tested.loc[:,["m_hodl", "k", "zeta_m", "zeta_v"]] = pd.Series(np.nan, index=range(0, len(dta_to_be_tested)))
        # ... recompute
        test_data = get_components(input_dta=dta_to_be_tested
                                , memory_param_m=inpt.loc[0, "c_m"]
                                , memory_param_v=inpt.loc[0, "c_v"]
                                , amplitude_param_m=inpt.loc[0, "q_m"]
                                , amplitude_param_v=inpt.loc[0, "q_v"]
                                , reset_logs=reset_logs
                                , dir_logging=dir_logging
                                , precision=float(0)
                                , verbose=False
                        )
        # ... compare
        test = test_data.reindex(sorted(test_data.columns), axis=1).round(5)
        true = true_data.reindex(sorted(true_data.columns), axis=1).round(5)
        test_results.append(test.equals(true))

    # --- Summarizing Tests
    all_passed = sum(test_results) == len(test_results)
    if all_passed:
        logger.info("ALL PASSED")
    else:
        logger.warning("NOT ALL TESTS PASSED")
    return all_passed


def get_fv(
            input_dta: dict
            , memory_param_m: float
            , memory_param_v: float
            , amplitude_param_m: float
            , amplitude_param_v: float
            , idx_start: int
            , induction_start: float
            , reset_logs: bool
            , dir_logging: str
            , precision=None
            , verbose=False
            , induction_correction=None):
    # ----- Setup Logging ----
    if dir_logging:
        # .. Thanks to https://docs.python.org/3/howto/logging-cookbook.html
        # Logging path handling
        path = pathlib.Path(dir_logging)
        if reset_logs and path.exists():
            shutil.rmtree(dir_logging)
        path.mkdir(parents=True, exist_ok=True)
        # create logger with 'spam_application'
        logger = logging.getLogger('_get_fv')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(dir_logging + "_get_fv.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        logger = None
    # ----- Prerequisites ----
    enum = range(idx_start, len(input_dta))
    input_dta.loc[1, "fv"] = induction_start
    if induction_correction:
        input_dta.loc[2, "fv"] = induction_correction # Note: only necessary for test_sheet (see testing loop)
    # ----- Loop ----
    # 1. Calc followings_day fv, write into databank -- repeat to end of time
    for t in enum:
        # Break loop if first fair value estimate is Nan due to k=1
        if t > 3:
            if np.isnan(input_dta.loc[t - 2, "prices"]):
                input_dta["prices"] = [np.nan]*len(input_dta)
                return input_dta;

        input_dta.loc[t - 1, "fv"] = fv_wrapped(
            dir_logging="./logs_fv/"
            , reset_logs=True
            , verbose=verbose
            , idx_t=t
            , vcirc_vec=input_dta["v_circ"]
            , onchain_txvol_usd=input_dta["onchain_vol_usd"]
            , supply_total=input_dta["m_total"]
            , time_vec=input_dta["time"]
            , price_vec=input_dta["prices"]
            , fv_vec=input_dta["fv"]
            , memory_param_v=memory_param_v
            , memory_param_m=memory_param_m
            , amplitude_param_m=amplitude_param_m
            , amplitude_param_v=amplitude_param_v
            , precision=precision
        )

    return input_dta


def get_theoretic_prices(
    input_dta: dict
        , memory_param_m: float
        , memory_param_v: float
        , amplitude_param_m: float
        , amplitude_param_v: float
        , reset_logs: bool
        , dir_logging: str
        , idx_start: int
        , precision: float
        , induction_start_prices_1: float
        , induction_start_prices_2:float
        , verbose=False
        , truncate_zH_until_period=False
        ):
    # ----- Setup Logging ----
    if dir_logging:
        # .. Thanks to https://docs.python.org/3/howto/logging-cookbook.html
        # Logging path handling
        path = pathlib.Path(dir_logging)
        if reset_logs and path.exists():
            shutil.rmtree(dir_logging)
        path.mkdir(parents=True, exist_ok=True)
        # create logger with 'spam_application'
        logger = logging.getLogger('_get_theoretic_prices')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(dir_logging + "_get_theoretic_prices.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        logger = None
    # ----- Prerequisites ----
    enum = range(idx_start, len(input_dta))
    input_dta.loc[0, "prices"] = induction_start_prices_1
    input_dta.loc[1, "prices"] = induction_start_prices_2
    # ----- Loop ----
    # 1. Calc followings_day fv, write into databank -- repeat to end of time
    for t in enum:

        # Break loop if first fair value estimate is Nan due to k=1
        if t > 3:
            if np.isnan(input_dta.loc[t - 2, "prices"]):
                input_dta["prices"] = [np.nan]*len(input_dta)
                return input_dta;

        input_dta.loc[t, "prices"] = theoretical_price_wrapped(
            dir_logging=None#"./logs_fv/"
            , reset_logs=True
            , verbose=verbose
            , idx_t=t
            , vcirc_vec=input_dta["v_circ"]
            , onchain_txvol_usd=input_dta["onchain_vol_usd"]
            , supply_total=input_dta["m_total"]
            , time_vec=input_dta["time"]
            , price_vec=input_dta["prices"]
            , fv_vec=input_dta["fv"]
            , memory_param_v=memory_param_v
            , memory_param_m=memory_param_m
            , amplitude_param_v=amplitude_param_v
            , amplitude_param_m=amplitude_param_m
            , precision=precision
            , truncate_zH_until_period=truncate_zH_until_period
        )
    if logger:
        logger.info("Finished simulation run: \n "
                "c_v = {} \n"
                "c_m = {} \n"
                "q_v  = {} \n"
                "q_m  = {} \n"
                "ind_start_prices_1 = {}"
                "ind_start_prices_1 = {}".format(memory_param_v
                                                  , memory_param_m
                                                  , amplitude_param_v
                                                  , amplitude_param_m
                                                  , induction_start_prices_1
                                                  , induction_start_prices_2
                                                  ))
    return input_dta


# ----- Setup Logging ----

def get_components(
            input_dta: dict
            , memory_param_m: float
            , memory_param_v: float
            , amplitude_param_m: float
            , amplitude_param_v: float
            , reset_logs: bool
            , dir_logging: str
            , verbose: bool
            , precision=None
            , truncate_zH_until_period=False
 ):
    # ----- Setup Logging ----
    if dir_logging:
        # .. Thanks to https://docs.python.org/3/howto/logging-cookbook.html
        # Logging path handling
        path = pathlib.Path(dir_logging)
        if reset_logs and path.exists():
            shutil.rmtree(dir_logging)
        path.mkdir(parents=True, exist_ok=True)
        # create logger with 'spam_application'
        logger = logging.getLogger('_get_components')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(dir_logging+"_get_components.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        logger = None
    # ----- Check for data ----
    if not input_dta["prices"].isnull()[1:len(input_dta)-1].sum() == 0 and \
            input_dta["prices"].isnull()[1:len(input_dta)-1].sum() == 0:
        raise ValueError('Use this function only for input data frames with complete FV and prices.')
    # ----- Prerequisites ----
    data_row_count=len(input_dta)
    enum = range(2, data_row_count)
    input_dta["zeta_m"] = pd.Series(np.nan, index=range(0, data_row_count))
    input_dta["zeta_v"] = pd.Series(np.nan, index=range(0, data_row_count))
    input_dta["k"] = pd.Series(np.nan, index=range(0, data_row_count))
    input_dta["m_hodl"] = pd.Series(np.nan, index=range(0, data_row_count))

    # ----- Loop ----
    for t in enum:
        input_dta.loc[t, "zeta_m"] = make_zeta(
                 logger=None
                 , verbose=False
                 , idx_t=t
                 , time_vec=input_dta["time"]
                 , price_vec=input_dta["prices"]
                 , fv_vec=input_dta["fv"]
                 , ignore_last_day=False
                 , zeta_type="momentum"
                 , memory_param=memory_param_m
                 , precision=precision
        )
        input_dta.loc[t,"zeta_v"] = make_zeta(
            logger=None
            , verbose=False
            , idx_t=t
            , time_vec=input_dta["time"]
            , price_vec=input_dta["prices"]
            , fv_vec=input_dta["fv"]
            , ignore_last_day=False
            , zeta_type="value"
            , memory_param=memory_param_v
            , precision=precision
        )
        input_dta.loc[t, "k"] = get_k_from_past(
            logger=None
            , verbose=False
            , idx_t=t
            , time_vec=input_dta["time"]
            , price_vec=input_dta["prices"]
            , fv_vec=input_dta["fv"]
            , memory_param_v=memory_param_v
            , memory_param_m=memory_param_m
            , amplitude_param_v=amplitude_param_v
            , amplitude_param_m=amplitude_param_m
            , ignore_last_day_zeta_v=False
            , precision=precision
            , truncate_zH_until_period=truncate_zH_until_period
        )
        input_dta.loc[t, "m_hodl"] = m_hodl_from_k(
            logger=None
            , verbose=False
            , idx_t=t
            , time_vec=input_dta["time"]
            , supply_total=input_dta["m_total"]
            , k_vec=input_dta["k"]
        )
    if dir_logging:
        logger.info("Finished run: \n "
                    "c_v = {} \n"
                    "c_m = {} \n"
                    "q_v = {} \n"
                    "q_m  = {}".format(memory_param_v
                                       , memory_param_m
                                       , amplitude_param_v
                                       , amplitude_param_m
                                       ))

    return input_dta

