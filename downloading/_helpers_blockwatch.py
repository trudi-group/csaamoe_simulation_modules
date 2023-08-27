# >>>>> General things >>>>> -------------------------------------
import time
import pandas as pd
from datetime import datetime
import importlib
import urllib.request
from pathlib import Path
import os
import yaml

path_settings = "./SETTINGS.yml"
path_gen_helpers = "./csaamoe_simulation_modules/gen_helpers.py"

# Import settings
with open(path_settings, 'r') as stream:
    try:
        SETTINGS = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Import general helpers
spec = importlib.util.spec_from_file_location("noname", path_gen_helpers)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def download_blockwatch_data(
        start
        , end
        , SETTINGS
        , limit=30000):
    # Setup input
    key=open(SETTINGS["downloads"]["blockwatch"]["api_key_path"], "r").read().replace("\n", "")
    # Form url
    url="https://data.blockwatch.cc/v1/series/" \
        "BTC-EOD/SUPPLY.csv?" \
        "start_date={}&" \
        "end_date={}&" \
        "limit={}&" \
        "api_key={}".format(start
                            , end
                            , limit
                            , key)

    # Load and make path if doesn't exist
    path = SETTINGS["downloads"]["blockwatch"]["path_to_save_csv_to"]
    ending = SETTINGS["downloads"]["blockwatch"]["eod_supply_name"]
    Path(path).mkdir(parents=True, exist_ok=True)

    # Download csv and rename
    helpers.blockPrint()
    urllib.request.urlretrieve(url, path+ ending)
    helpers.enablePrint()


def load_blockwatch_data(
        SETTINGS):
    # Load data
    path = SETTINGS["downloads"]["blockwatch"]["path_to_save_csv_to"]
    ending = SETTINGS["downloads"]["blockwatch"]["eod_supply_name"]
    dta = pd.read_csv(path+ending)
    return dta


def standardize_blockwatch_data(
        SETTINGS):
    # Load data
    dta = load_blockwatch_data(SETTINGS)
    # Date (form millisec to sec)
    dta["time"] = [datetime.utcfromtimestamp(int(i/1000)).strftime('%Y-%m-%d') for i in dta["time"]]
    # Save
    dta.to_csv(SETTINGS["downloads"]["blockwatch"]["path_to_save_csv_to"]+
               SETTINGS["downloads"]["blockwatch"]["eod_supply_name"])


def get_blockwatch_data(start,
                        end,
                        SETTINGS,
                        force_update=False):
    condition_file  = os.path.isfile(SETTINGS["downloads"]["blockwatch"]["path_to_save_csv_to"] + SETTINGS["downloads"]["blockwatch"]["eod_supply_name"])
    condition_force_update = not force_update
    if condition_force_update or condition_file:
        dta=load_blockwatch_data(SETTINGS)
    else:
        download_blockwatch_data(start, end, SETTINGS)
        standardize_blockwatch_data(SETTINGS)
        dta=load_blockwatch_data(SETTINGS)

    condition_dates = not start in dta["time"] or end in start in dta["time"]
    if condition_dates:
        download_blockwatch_data(start, end, SETTINGS)
        standardize_blockwatch_data(SETTINGS)
        dta = load_blockwatch_data(SETTINGS)
    return dta