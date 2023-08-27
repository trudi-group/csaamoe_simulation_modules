# >>>>> General things >>>>> -------------------------------------
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import *
from functools import reduce
import importlib

# Import general helpers
path_settings = "./SETTINGS.yml"
path_gen_helpers = "./csaamoe_simulation_modules/gen_helpers.py"
spec = importlib.util.spec_from_file_location("noname", "./csaamoe_simulation_modules/gen_helpers.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def get_coins_markets_paginated(full_request=1000,
                                answers_per_page=250):
    if 0 != (full_request % answers_per_page):
        raise Exception("Input 'full_request' not divisible by 'answers_per_page'. ")

    pages = int(full_request / answers_per_page)
    coininfo = list()
    for page in tqdm(range(pages)):
        coininfo_sub = cg.get_coins_markets(vs_currency="USD",
                                            per_page=250,
                                            page=page)
        coininfo += coininfo_sub

        time.sleep(3)

    return coininfo

#
# def get_sifted_coin_ids(req_nbr=SETTINGS["downloads"]["coingecko"]["full_request_number"],
#                         answ_p_page=SETTINGS["downloads"]["coingecko"]["answers_per_page"]):
#     info_raw = get_coins_markets_paginated(full_request=req_nbr,
#                                            answers_per_page=answ_p_page)
#     ids = [info_raw[i]["id"] for i in range(len(info_raw))]
#     ids_unique = list(set(ids))
#
#     return ids_unique
#


def get_coingecko_data(SETTINGS,
                         coinlist,
                         cg):
    # prepare loop
    dta_succ_ids = list()
    dta_gathered = list()

    # split timeperiod into sub-periods
    if not SETTINGS["downloads"]["update"]:
        enddate = datetime.fromtimestamp(SETTINGS["downloads"]["timestamp_to"])
    else:
        enddate = datetime.utcnow()
    startdate = datetime.fromtimestamp(SETTINGS["downloads"]["timestamp_from"])
    subperiods = [round((startdate + relativedelta(months=i)).timestamp())
                  for i in range(round((enddate - startdate).days / 30))]

    for id in tqdm(coinlist):
        prices = pd.DataFrame()
        market_caps = pd.DataFrame()
        total_volumes = pd.DataFrame()
        for subp in range(len(subperiods) - 1):
            # Economize on rate limit
            time.sleep(1)
            # Prep start- and end-timestamp
            raw_packed = None
            from_sub_ts = subperiods[subp]
            to_sub_ts = subperiods[subp + 1] - 1
            while raw_packed is None:
                try:
                    raw_packed = cg.get_coin_market_chart_range_by_id(id=id,
                                                                      from_timestamp=from_sub_ts,
                                                                      to_timestamp=to_sub_ts,
                                                                      vs_currency=['usd'])
                except:
                    pass
            if (len(raw_packed["prices"]) > 0
                    or len(raw_packed["total_volumes"])
                    or len(raw_packed["market_caps"])):

                prices_sub = pd.DataFrame(list(zip(*raw_packed["prices"]))).T
                prices_sub[0] = pd.to_datetime(prices_sub[0], unit="ms").dt.floor("H")

                total_volumes_sub = pd.DataFrame(list(zip(*raw_packed["total_volumes"]))).T
                total_volumes_sub[0] = pd.to_datetime(total_volumes_sub[0], unit="ms").dt.floor("H")

                market_caps_sub = pd.DataFrame(list(zip(*raw_packed["market_caps"]))).T
                market_caps_sub[0] = pd.to_datetime(market_caps_sub[0], unit="ms").dt.floor("H")

                prices = prices.append(prices_sub, ignore_index=True)

                total_volumes = total_volumes.append(total_volumes_sub, ignore_index=True)

                market_caps = market_caps.append(market_caps_sub, ignore_index=True)
                dta_list = [prices, total_volumes, market_caps]
                dta = reduce(lambda left, right: pd.merge(left, right, on=0, how="outer"),
                             dta_list)  # "0" is alway timestamp-column
                dta.columns = ["time", "prices", "total_volumes", "market_caps"]
                # Build list of dataframes from indiv. download
                dta_gathered.append(dta)
                dta_succ_ids.append(id)

    # Make dict from list
    dta_gathered = helpers.list_to_dictonary(names=dta_succ_ids,datalist=dta_gathered)
    return dta_gathered

