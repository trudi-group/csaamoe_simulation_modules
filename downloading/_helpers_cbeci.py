# >>>>> General things >>>>> -------------------------------------
import time
import pandas as pd
from datetime import datetime
import importlib
import pandas as pd
import datetime
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


def get_cbeci_sha_data(SETTINGS):
    # Load data
    print("WARNING: Note, that the CBECI needs to updated manually.")
    path = SETTINGS["downloads"]["cbeci"]["path_to_save_csv_to"]
    ending = SETTINGS["downloads"]["cbeci"]["cbeci_sha_list_name"]
    dta = pd.read_csv(path+ending)
    # some editing
    print("WARNING: Note, that from the CBECI, we use the Unix timestamps and correct inconsistent dates.")
    dta["Date of release"] = [datetime.datetime.fromtimestamp(d).strftime("%Y-%m-%d") for d in dta["UNIX_date_of_release"]]
    return dta


def transform_cbeci_sha_data(dta
                             , day_start
                             , day_end
                             , aggtype):
    # Create date vector between start and end date
    dates = pd.date_range(pd.to_datetime(day_start, format="%Y-%m-%d")
                  , pd.to_datetime(day_end, format="%Y-%m-%d") - datetime.timedelta(days=1)
                  , freq='d').strftime("%Y-%m-%d")
    dta_date = pd.DataFrame(dates, columns=["time"])
    # for same release date, use more or less efficient hardware
    if aggtype == "low":
        dta = dta.groupby("Date of release", as_index=False).max()
    elif aggtype == "high":
        dta = dta.groupby("Date of release", as_index=False).min()
    # merge in the data into time vector
    d = pd.merge(dta_date, dta, left_on="time", right_on="Date of release", how="left")
    d.fillna(method='ffill', inplace=True)

    return d
