import pandas as pd
import logging
import shutil
import pathlib

from . _components import make_zeta, get_k_from_past, get_implied_k, fv, theoretical_price
from . _utils import test_diff, resample_to_daily


def testing_core_model(
    dir_test_data
    , reset_logs: bool
    , dir_logging
    , fname_test_data
    , precision
):
    # ----- Setup Logging ----
    # .. Thanks to https://docs.python.org/3/howto/logging-cookbook.html
    # Logging path handlling
    path = pathlib.Path(dir_logging)
    if reset_logs and path.exists():
        shutil.rmtree(dir_logging)
    path.mkdir(parents=True, exist_ok=True)
    # -- General logger (test results)
    # create logger
    logger = logging.getLogger('test_results_core_model')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(dir_logging+"test_results_core_model.log")
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
    # -- Lower level loggers
    logger_zeta_m = logging.getLogger('_zeta_m')
    logger_zeta_m.setLevel(logging.DEBUG)
    fh = logging.FileHandler(dir_logging+"__testing_zeta_m.log")
    fh.setLevel(logging.DEBUG)
    logger_zeta_m.addHandler(fh)
    # -
    logger_zeta_v = logging.getLogger('__testing_zeta_v')
    logger_zeta_v.setLevel(logging.DEBUG)
    fh = logging.FileHandler(dir_logging+"__testing_zeta_v.log")
    fh.setLevel(logging.DEBUG)
    logger_zeta_v.addHandler(fh)
    # -
    logger_zeta_v_ign = logging.getLogger('_zeta_v_ign')
    logger_zeta_v_ign.setLevel(logging.DEBUG)
    fh = logging.FileHandler(dir_logging+"__testing_zeta_v_ign.log")
    fh.setLevel(logging.DEBUG)
    logger_zeta_v_ign.addHandler(fh)
    # -
    logger_k = logging.getLogger('_k')
    logger_k.setLevel(logging.DEBUG)
    fh = logging.FileHandler(dir_logging+"__testing_k.log")
    fh.setLevel(logging.DEBUG)
    logger_k.addHandler(fh)
    # -
    logger_k_ign = logging.getLogger('_k_ign')
    logger_k_ign.setLevel(logging.DEBUG)
    fh = logging.FileHandler(dir_logging+"__testing_k_ign.log")
    fh.setLevel(logging.DEBUG)
    logger_k_ign.addHandler(fh)
    # -
    logger_k_implied = logging.getLogger('_k_implied')
    logger_k_implied.setLevel(logging.DEBUG)
    fh = logging.FileHandler(dir_logging+"__testing_k_implied.log")
    fh.setLevel(logging.DEBUG)
    logger_k_implied.addHandler(fh)
    # -
    logger_fv = logging.getLogger('_fv')
    logger_fv.setLevel(logging.DEBUG)
    fh = logging.FileHandler(dir_logging+"__testing_fv.log")
    fh.setLevel(logging.DEBUG)
    logger_fv.addHandler(fh)

    # Load test data
    logger.info("Loading test-data from: " + dir_test_data)
    test_dta_raw = resample_to_daily(pd.read_csv(dir_test_data+fname_test_data), "time")
    test_input = {"dta": test_dta_raw.loc[:, ["time"
                                                 , "prices"
                                                 , "v_circ"
                                                 , "fv"
                                                 , "onchain_vol_usd"
                                                 , "m_total"
                                                 , "m_hodl"
                                                 , "k"
                                                 , "k_implied"
                                                 , "k_ign"
                                                 , "zeta_v_ign"
                                                 , "zeta_m"
                                                 , "zeta_v"
                                              ]]
        , "q_v": test_dta_raw.loc[0, "q_v"]
        , "q_m": test_dta_raw.loc[0, "q_m"]
        , "c_v": test_dta_raw.loc[0, "c_v"]
        , "c_m": test_dta_raw.loc[0, "c_m"]}

    # Test data is in lines 4 to 8 of the testdata-csv
    all_test_results = {}
    for t in range(4, 8):
        allowed_precision_loss = 10 * 0.5
        rdigits = int(allowed_precision_loss)
        zeta_m = make_zeta(
            logger=logger_zeta_m
            , verbose=False
            , idx_t=t
            , time_vec=test_input["dta"]["time"]
            , price_vec=test_input["dta"]["prices"]
            , fv_vec=test_input["dta"]["fv"]
            , ignore_last_day=False
            , zeta_type="momentum"
            , memory_param=test_input["c_m"]
            , precision=precision
        )
        test_result = test_diff(val_test=zeta_m
                            , val_true=test_input["dta"]["zeta_m"][t]
                            , digits=rdigits)
        logger.info("zeta_m: case:{} , {}".format(str(t), test_result))

        zeta_v = make_zeta(
            logger=logger_zeta_v
            , verbose=False
            , idx_t=t
            , time_vec=test_input["dta"]["time"]
            , price_vec=test_input["dta"]["prices"]
            , fv_vec=test_input["dta"]["fv"]
            , ignore_last_day=False
            , zeta_type="value"
            , memory_param=test_input["c_v"]
            , precision=precision
        )
        test_result = test_diff(val_test=zeta_v
                                        , val_true=test_input["dta"]["zeta_v"][t]
                                        , digits=rdigits)
        logger.info("zeta_v: case:{} , {}".format(str(t), test_result))

        zeta_v_ign = make_zeta(
            logger=logger_zeta_v_ign
            , verbose=False
            , idx_t=t
            , time_vec=test_input["dta"]["time"]
            , price_vec=test_input["dta"]["prices"]
            , fv_vec=test_input["dta"]["fv"]
            , ignore_last_day=True
            , zeta_type="value"
            , memory_param=test_input["c_v"]
            , precision=precision
        )
        test_result = test_diff(val_test=zeta_v_ign
                                          , val_true=test_input["dta"]["zeta_v_ign"][t]
                                          , digits=rdigits)
        logger.info("zeta_v_ign: case:{} , {}".format(str(t), test_result))

        k = get_k_from_past(
            logger=logger_k
            , verbose=False
            , idx_t=t
            , time_vec=test_input["dta"]["time"]
            , price_vec=test_input["dta"]["prices"]
            , fv_vec=test_input["dta"]["fv"]
            , memory_param_v=test_input["c_v"]
            , memory_param_m=test_input["c_m"]
            , amplitude_param_v=test_input["q_v"]
            , amplitude_param_m=test_input["q_m"]
            , ignore_last_day_zeta_v=False
            , precision=precision
        )
        test_result = test_diff(val_test=k
                                         , val_true=test_input["dta"]["k"][t]
                                         , digits=rdigits)
        logger.info("k: case:{} , {}".format(str(t), test_result))

        k_ign = get_k_from_past(
            logger=logger_k_ign
            , verbose=False
            , idx_t=t
            , time_vec=test_input["dta"]["time"]
            , price_vec=test_input["dta"]["prices"]
            , fv_vec=test_input["dta"]["fv"]
            , memory_param_v=test_input["c_v"]
            , memory_param_m=test_input["c_m"]
            , amplitude_param_v=test_input["q_v"]
            , amplitude_param_m=test_input["q_m"]
            , ignore_last_day_zeta_v=True
            , precision=precision
        )
        test_result = test_diff(val_test=k_ign
                                         , val_true=test_input["dta"]["k_ign"][t]
                                         , digits=rdigits)
        logger.info("k_ign: case:{} , {}".format(str(t), test_result))

        k_implied = get_implied_k(
            logger=logger_k_implied
            , verbose=False
            , idx_t=t
            , time_vec=test_input["dta"]["time"]
            , price_vec=test_input["dta"]["prices"]
            , vcirc_vec=test_input["dta"]["v_circ"]
            , onchain_txvol_usd=test_input["dta"]["onchain_vol_usd"]
            , supply_total=test_input["dta"]["m_total"]
        )
        test_result = test_diff(val_test=k_implied
                                         , val_true=test_input["dta"]["k_implied"][t]
                                         , digits=rdigits)
        logger.info("k_implied: case:{} , {}".format(str(t), test_result))

        val = fv(
            logger=logger_fv
            , verbose=False
            , idx_t=t
            , time_vec=test_input["dta"]["time"]
            , price_vec=test_input["dta"]["prices"]
            , vcirc_vec=test_input["dta"]["v_circ"]
            , fv_vec=test_input["dta"]["fv"]
            , onchain_txvol_usd=test_input["dta"]["onchain_vol_usd"]
            , supply_total=test_input["dta"]["m_total"]
            , memory_param_v=test_input["c_v"]
            , memory_param_m=test_input["c_m"]
            , amplitude_param_v=test_input["q_v"]
            , amplitude_param_m=test_input["q_m"]
            , precision=precision
        )
        test_result = test_diff(val_test=val
                                         , val_true=test_input["dta"]["fv"][t-1]
                                         , digits=rdigits)
        logger.info("FV backed out: case:{} , {}".format(str(t), test_result))

    # Summarizing Tests
    all_tests_passed = all(["passed" == result for result in all_test_results.values()])
    if all_tests_passed:
        logger.info("ALL PASSED")
    else:
        logger.warning("NOT ALL TESTS PASSED")
    return all_tests_passed


def fv_wrapped(
        dir_logging
        , reset_logs
        , verbose
        , idx_t
        , time_vec
        , price_vec
        , vcirc_vec
        , fv_vec
        , onchain_txvol_usd
        , supply_total
        , memory_param_v
        , memory_param_m
        , amplitude_param_v
        , amplitude_param_m
        , precision
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
        logger = logging.getLogger('pricemodels')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(dir_logging+"_fv.log")
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
    # ----- Run ------
    val = fv(
            logger=logger
            , verbose=verbose
            , idx_t=idx_t
            , time_vec=time_vec
            , price_vec=price_vec
            , vcirc_vec=vcirc_vec
            , fv_vec=fv_vec
            , onchain_txvol_usd=onchain_txvol_usd
            , supply_total=supply_total
            , memory_param_v=memory_param_v
            , memory_param_m=memory_param_m
            , amplitude_param_v=amplitude_param_v
            , amplitude_param_m=amplitude_param_m
            , precision=precision
        )
    return val


def theoretical_price_wrapped(
        dir_logging: str
        , reset_logs: bool
        , verbose: bool
        , idx_t: int
        , time_vec: list
        , price_vec: list
        , vcirc_vec: list
        , fv_vec: list
        , onchain_txvol_usd: list
        , supply_total: list
        , memory_param_v: float
        , memory_param_m: float
        , amplitude_param_v: float
        , amplitude_param_m: float
        , precision: float
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
        logger = logging.getLogger('pricemodels')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(dir_logging+"_theoretical_price.log")
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

    # ----- Run ------
    price = theoretical_price(
            logger=logger
            , verbose=verbose
            , idx_t=idx_t
            , time_vec=time_vec
            , price_vec=price_vec
            , vcirc_vec=vcirc_vec
            , fv_vec=fv_vec
            , onchain_txvol_usd=onchain_txvol_usd
            , supply_total=supply_total
            , memory_param_v=memory_param_v
            , memory_param_m=memory_param_m
            , amplitude_param_v=amplitude_param_v
            , amplitude_param_m=amplitude_param_m
            , precision=precision
            , truncate_zH_until_period=truncate_zH_until_period
        )
    return price


