import sys
import yaml
import importlib
from datetime import datetime
from pycoingecko import CoinGeckoAPI

sys.path.append('./csaamoe_simulation_modules/')
from datalogistics import load_db
from . _helpers_coingecko  import get_coingecko_data
from . _helpers_quandl  import get_quandl_data
from . _helpers_coindesk  import get_coindesk_data
from . _helpers_blockwatch  import get_blockwatch_data
from . _helpers_cbeci import get_cbeci_sha_data, transform_cbeci_sha_data


# >>>>> General things >>>>> -------------------------------------
def collect_raw_data(path_settings, path_gen_helpers):
    # Import Config
    with open(path_settings, 'r') as stream:
        try:
            SETTINGS = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Import general helpers
    spec = importlib.util.spec_from_file_location("noname", path_gen_helpers)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    # -- Set download period --------------------------------------------------
    if not SETTINGS["downloads"]["update"]:
        day_end = datetime.fromtimestamp(SETTINGS["downloads"]["timestamp_to"]).strftime("%Y-%m-%d")
    else:
        day_end = datetime.utcnow().strftime("%Y-%m-%d")
    day_start = datetime.fromtimestamp(SETTINGS["downloads"]["timestamp_from"]).strftime("%Y-%m-%d")

    # # >>>>> COINGECKO Download data >>>>> -------------------------------------
    # # From: coingecko.com
    # # Download mkt data
    # ids = SETTINGS["downloads"]["coingecko"]["ids"].split(", ")
    # dta = get_coingecko_data(SETTINGS=SETTINGS
    #                         , coinlist= ids
    #                         , cg=CoinGeckoAPI()
    # )
    # # Save data
    # connection = load_db(db_path=SETTINGS["downloads"]["db_dir"], db_name=SETTINGS["downloads"]["db_name"])
    # for i in ids:
    #     dta[i].to_sql(name="{}_coingecko".format(i)
    #                , con=connection
    #                , index=False
    #                , if_exists="replace")

    # >>>>> QUANDL Download data >>>>> -------------------------------------
    # From: quandl.com & blockchain.info as original source
    # Download
    dta = get_quandl_data(day_start=day_start
                            , day_end=day_end
                            , SETTINGS=SETTINGS)
    # Save data
    connection = load_db(db_path=SETTINGS["downloads"]["db_dir"], db_name=SETTINGS["downloads"]["db_name"])
    dta.to_sql(name= "bitcoin_quandl"
                , con=connection
                , index=False
                , if_exists="replace")

    # >>>>> Coindesk BTC BPI Download data >>>>> -------------------------------------
    # Download
    dta = get_coindesk_data(day_start
                            , day_end)
    # Save data
    connection = load_db(db_path=SETTINGS["downloads"]["db_dir"], db_name=SETTINGS["downloads"]["db_name"])
    dta.to_sql(name= "bitcoin_coindesk_bpi"
                , con=connection
                , index=False
                , if_exists="replace")

    # >>>>> Blockwatch Download data >>>>> -------------------------------------
    # Download
    dta = get_blockwatch_data(day_start
                              , day_end
                              , SETTINGS)
    # Save data
    connection = load_db(db_path=SETTINGS["downloads"]["db_dir"], db_name=SETTINGS["downloads"]["db_name"])
    dta.to_sql(name= "bitcoin_blockwatch_supply_stats"
                , con=connection
                , index=False
                , if_exists="replace")
    # >>>>> CBECI data >>>>> -------------------------------------
    # Download
    dta = get_cbeci_sha_data(SETTINGS)
    dta_transformed_high = transform_cbeci_sha_data(dta
                                                , day_start
                                                , day_end
                                                , aggtype="high")
    dta_transformed_low = transform_cbeci_sha_data(dta
                                                    , day_start
                                                    , day_end
                                                    , aggtype="low")

    # Save data
    connection = load_db(db_path=SETTINGS["downloads"]["db_dir"], db_name=SETTINGS["downloads"]["db_name"])
    dta_transformed_high.to_sql(name= "bitcoin_cbeci_sha_high"
                                , con=connection
                                , index=False
                                , if_exists="replace")
    dta_transformed_low.to_sql(name= "bitcoin_cbeci_sha_low"
                                , con=connection
                                , index=False
                                , if_exists="replace")
    # # >>>>> CryptoQuant data >>>>> -------------------------------------
    # # Download
    # dta = get_cryptoquant_data(day_start
    #                           , day_end
    #                           , SETTINGS)
    # # Save data
    # connection = load_db(db_path=SETTINGS["downloads"]["db_dir"], db_name=SETTINGS["downloads"]["db_name"])
    # dta.to_sql(name= "bitcoin_cryptoquant_data"
    #             , con=connection
    #             , index=False
    #             , if_exists="replace")
