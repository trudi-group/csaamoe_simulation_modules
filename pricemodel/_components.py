import numpy as np
import pandas as pd


def make_zeta(
    logger
    , verbose: bool
    , time_vec: list
    , price_vec: list
    , idx_t: int
    , ignore_last_day: bool
    , zeta_type: str
    , memory_param: float
    , precision: float
    , fv_vec=None
):

    # Preparing Tuples for Zeta
    tuple_selection_for_period = list()

    # Catch if Zeta is calculated as "tilde"-version
    if not ignore_last_day:
        tuple_number = idx_t
    else:
        tuple_number = idx_t - 1  # -1 as don't need element (t) in the sum

    for idx_tuple in range(0, tuple_number):
        if idx_tuple - 1 >= 0:
            tuple_selection_for_period.append([idx_tuple - 1, idx_tuple])

    # --- Create Zeta
    zeta = 0
    how_far_in_past_vec = list()
    weights = list()
    summands = list()
    day_t = None
    summand_dates = list()
    # The difference between "momentum" and "value" zetas, apart from the
    # .. obvious theory, is that "value" needs only the variable of day t-1 for
    # .. the zeta calculated for S(t). That is - we need to start with the
    # .. calc. 1 step earlier if we want to use the same information set for
    # .. both zetas. Refer to the testdata.odt for seeing this.
    enumerator = enumerate(tuple_selection_for_period)
    for idx, idx_tuple in enumerator:
        # ----- Setting vars -----
        # Components of first factor of sum
        idx_sum_end = (idx_t - 1)
        how_far_in_past = idx_sum_end - (idx + 1) # "+1" as 0-indexing in python
        weight = np.exp(-how_far_in_past / memory_param)

        # ---------- Break if weight very low  ----------
        if abs(weight) < precision:
            continue
        # ---------- Sum ----------
        if zeta_type == "momentum":
            # Components of second factor of sum
            price_updated = price_vec[idx_tuple[1]]  # "larger" time if it was timestamp
            price_anchor = price_vec[idx_tuple[0]]  # "smaller" time if it was timestamp
            date_of_updated = time_vec[idx_tuple[1]]  # "larger" time if it was timestamp
            date_of_anchor = time_vec[idx_tuple[0]]  # "smaller" time if it was timestamp
            # Sum
            summand = (price_updated - price_anchor) / price_anchor
        elif zeta_type == "value":
            # Components of second factor of sum
            price_updated = price_vec[idx_tuple[1]]  # "smaller" time if it was timestamp
            date_at_update = time_vec[idx_tuple[1]]  # "smaller" time if it was timestamp
            fv_at_update = fv_vec[idx_tuple[1]]
            # Sum
            summand = (fv_at_update - price_updated) / price_updated
        else:
            raise Exception('Unspecified function input.')
        zeta += weight * summand

        # ------ For checking -----
        day_t = time_vec[idx_t].strftime("%m/%d/%Y")
        how_far_in_past_vec.append(how_far_in_past)
        weights.append(round(weight, 4))
        if zeta_type == "momentum":
            summand_dates.append('({} - {}) / {}'.format(
                date_of_updated.strftime("%m/%d/%Y")
                , date_of_anchor.strftime("%m/%d/%Y")
                , date_of_anchor.strftime("%m/%d/%Y")))
            summands.append('({} - {}) / {}'.format(
                round(price_updated, 4)
                , round(price_anchor, 4)
                , round(price_anchor, 4)))
        elif zeta_type == "value":
            summand_dates.append('{}'.format(
                date_at_update.strftime("%m/%d/%Y")))
            summands.append('({} - {}) / {}'.format(
                round(fv_at_update, 4)
                , round(price_updated, 4)
                , round(price_updated, 4)))

    logger_input = "----------------------------------------------\n" \
                     "---   Get k from history for Index(t)={}\n" \
                     "---           Type = << {} >> \n" \
                     "Setup: Memory param: {}\n" \
                     "Setup: Date t set to: {}\n" \
                     "> Summands at: {}\n" \
                     "> Summands: {}\n" \
                     "> Weights: {}\n" \
                     "> Result: {}".format(idx_t
                                       , zeta_type
                                       , memory_param
                                       , day_t
                                       , summand_dates
                                       , summands
                                       , weights
                                       , round(zeta, 4))
    if logger:
        logger.debug(logger_input)
    if verbose:
        print(logger_input)
    # -----------------------------------------------------------------------

    return zeta


def get_k_from_past(
    logger: object
    , verbose: bool
    , memory_param_v: float
    , memory_param_m: float
    , amplitude_param_v: float
    , amplitude_param_m: float
    , time_vec: list
    , price_vec: list
    , idx_t: int
    , ignore_last_day_zeta_v: bool
    , precision: float
    , fv_vec=None
    , truncate_zH_until_period=False
):
    zeta_m = make_zeta(
        logger=None
        , verbose=verbose
        , price_vec=price_vec
        , time_vec=time_vec
        , memory_param=memory_param_m
        , idx_t=idx_t
        , zeta_type="momentum"
        , ignore_last_day=False
        , precision=precision
    )

    zeta_v = make_zeta(
        logger = None
        , verbose=verbose
        , price_vec=price_vec
        , time_vec=time_vec
        , memory_param=memory_param_v
        , idx_t=idx_t
        , zeta_type="value"
        , ignore_last_day=ignore_last_day_zeta_v
        , fv_vec=fv_vec
        , precision=precision
        )

    # k = 0.5 + \
    #     0.5 * (amplitude_param_m/memory_param_m)*zeta_m + \
    #     0.5 * (amplitude_param_v/memory_param_v)*zeta_v
    #

    k = 0.5 + \
       0.5 * np.tanh((amplitude_param_m/memory_param_m)*zeta_m + (amplitude_param_v/memory_param_v)*zeta_v)

    # #k = 0.5 * (amplitude_param_m/memory_param_m)*zeta_m + \
    #     0.5 * (amplitude_param_v/memory_param_v)*zeta_v

    # block momentum until a cetain point
    if truncate_zH_until_period and idx_t < truncate_zH_until_period:
        k = 0.5 + 0.5 * np.tanh((0/memory_param_m)*zeta_m + (amplitude_param_v/memory_param_v)*zeta_v)

    if k > 1:
        k = 1
    if k < 0:
        k = 0

    if logger:
        logger.debug("Not much to log here. See zeta-logs.")

    return k


def get_implied_k(
    logger: object
    , verbose: bool
    , idx_t: int
    , time_vec: list
    , price_vec: list
    , vcirc_vec: list
    , onchain_txvol_usd: list
    , supply_total: list
):
    # k_implied = (T_t/V_t)/(S_t*M_tm1) - M_t/M_tm1
    summand_1 = supply_total[idx_t]/supply_total[idx_t-1]
    summand_2 = ( onchain_txvol_usd[idx_t] / vcirc_vec[idx_t] ) / ( price_vec[idx_t] * supply_total[idx_t-1] )
    k_implied = summand_1 - summand_2

    logger_input = "----------------------------------------------\n" \
                     "---     Get implied k for Index(t)={}\n" \
                     "Setup: Date t set to: {}\n" \
                     "> Summand 1: M_({})/M_({})\n" \
                     "> Summand 2: (T*_({})/V*_({})) / (S_({}) M_({}))\n" \
                     "> Result: {}".format(idx_t
                                            , time_vec[idx_t].strftime("%m/%d/%Y")
                                            , time_vec[idx_t].strftime("%m/%d/%Y")
                                            , time_vec[idx_t-1].strftime("%m/%d/%Y")
                                            , time_vec[idx_t].strftime("%m/%d/%Y")
                                            , time_vec[idx_t].strftime("%m/%d/%Y")
                                            , time_vec[idx_t].strftime("%m/%d/%Y")
                                            , time_vec[idx_t-1].strftime("%m/%d/%Y")
                                            , round(k_implied, 2))

    if logger:
        logger.debug(logger_input)
    if verbose:
        print(logger_input)

    return k_implied


def fv(logger: object
       , verbose: bool
       , idx_t: int
        , time_vec: list
        , price_vec: list
        , vcirc_vec
        , fv_vec
        , onchain_txvol_usd: list
        , supply_total: list
        , memory_param_v: float
        , memory_param_m: float
        , amplitude_param_m: float
        , amplitude_param_v: float
       , precision
):
    # ------ For logging
    logger_input = "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n" \
                     "Backing out FV for \n " \
                     " -> Period (t-1) = {} that is \n" \
                     " -> Index(t-1) = {}.  ".format(time_vec[idx_t - 1].strftime("%m/%d/%Y")
                                                     ,idx_t - 1)
    if logger:
        logger.debug(logger_input)

    # ------ Prepare input
    price_lag = price_vec[idx_t - 1]
    ratio_ign = memory_param_v / amplitude_param_v
    k_implied = get_implied_k(
                                logger=None
                                , verbose=verbose
                                , idx_t=idx_t
                                , time_vec=time_vec
                                , price_vec=price_vec
                                , vcirc_vec=vcirc_vec
                                , onchain_txvol_usd=onchain_txvol_usd
                                , supply_total=supply_total
    )
    k_ign = get_k_from_past(
        logger=None
        , verbose=verbose
        , memory_param_v=memory_param_v
        , memory_param_m=memory_param_m
        , amplitude_param_v=amplitude_param_v
        , amplitude_param_m=amplitude_param_m
        , time_vec=time_vec
        , price_vec=price_vec
        , fv_vec=fv_vec
        , idx_t=idx_t
        , ignore_last_day_zeta_v=True
        , precision=precision
        , truncate_zH_until_period=False
    )

    # ----- Back out fundamental value
    # Break case: if k = 1 this leads to distortions as k is determined by a dis-continous function
    if k_ign == 1:
        return np.nan;
    else:
        diff_ign = k_implied - k_ign
        fv_lag = price_lag * (1 + 2 * ratio_ign * diff_ign)
        return fv_lag;


def m_hodl_from_k(logger: object
        , verbose: bool
        , idx_t: int
        , time_vec: list
        , supply_total: list
        , k_vec: list
):
    # ------ For logging
    logger_input = "------------- \n" \
                     "Getting M held by speculators from k as  \n " \
                     "M_{}/k_{}".format(time_vec[idx_t].strftime("%m/%d/%Y")
                                                     ,time_vec[idx_t-1].strftime("%m/%d/%Y"))
    if logger:
        logger.debug(logger_input)
    if verbose:
        print(logger_input)

    # k_implied = (T_t/V_t)/(S_t*M_tm1) - M_t/M_tm1
    m_hodl = supply_total[idx_t-1]*k_vec[idx_t]

    return m_hodl


def theoretical_price(logger: object
        , verbose: bool
        , idx_t: int
        , time_vec: list
        , price_vec: list
        , vcirc_vec
        , fv_vec
        , precision: float
        , onchain_txvol_usd: list
        , supply_total: list
        , memory_param_v: float
        , memory_param_m: float
        , amplitude_param_v: float
        , amplitude_param_m: float
        , truncate_zH_until_period=False
        ):
    # ------ For logging
    logger_input = "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n" \
                     "Backing out FX (Market exchange rate) for \n " \
                     " -> Period (t) = {} that is \n" \
                     " -> Index(t) = {}.  ".format(time_vec[idx_t].strftime("%m/%d/%Y")
                                                     ,idx_t)
    if logger:
        logger.debug(logger_input)

    # ------ Prepare input
    k = get_k_from_past(
        logger=None
        , verbose=verbose
        , memory_param_v=memory_param_v
        , memory_param_m=memory_param_m
        , amplitude_param_v=amplitude_param_v
        , amplitude_param_m=amplitude_param_m
        , time_vec=time_vec
        , price_vec=price_vec
        , fv_vec=fv_vec
        , idx_t=idx_t
        , ignore_last_day_zeta_v=False
        , precision=precision
        , truncate_zH_until_period=truncate_zH_until_period
    )

    m_total = supply_total[idx_t]
    m_total_lag = supply_total[idx_t-1]
    v_circ = vcirc_vec[idx_t]
    onchain_vol = onchain_txvol_usd[idx_t]
    # ----- Calc. market fx
    m_circ = m_total - k*m_total_lag
    theoretical_price = (onchain_vol/v_circ)/m_circ

    if k == 1:
        return np.nan;
    else:
        return theoretical_price;
