import matplotlib.pyplot as plt
import pickle
import yaml
from datetime import datetime
import os
import sys

# Import Config
with open("./SETTINGS.yml", 'r') as stream:
    try:
        SETTINGS = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# sub-function transforming a timestamp as returned by the CoinGeckoAPI to a string date
def custom_unixts_to_date(ts,
                          in_milli_sec=True):
    if in_milli_sec:
        ts_clean = int(ts / 1000)
    else:
        ts_clean = int(ts)

    d = datetime.fromtimestamp(ts_clean).strftime(format="%Y-%m-%d")
    return d


def save_to_csv(config
            , d
            , table_name
            , path_csv=None):
    if not SETTINGS["downloads"]["update"]:
        startdate_raw = datetime.fromtimestamp(SETTINGS["downloads"]["timestamp_from"])
    else:
        startdate_raw = round(datetime.utcnow().timestamp())
    enddate_raw = datetime.fromtimestamp(SETTINGS["downloads"]["timestamp_to"])

    startdate = custom_unixts_to_date(startdate_raw,
                                      in_milli_sec=False)
    enddate = custom_unixts_to_date(enddate_raw,
                                    in_milli_sec=False)

    if not os.path.exists(path_csv):
        os.makedirs(path_csv)

    for nme in d.keys():
        d[nme].to_csv(path_or_buf=(path_csv
                                   + table_name
                                   + "_"
                                   + startdate
                                   + "_"
                                   + enddate
                                   + ".csv"),
                      index=False)


def roundTime(dt=None, roundTo=60):
    """Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    if dt == None: dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


def list_to_dictonary(names, datalist):
    # Create a dictionary from zip object
    list_zip   = zip(names, datalist)
    list_dict  = dict(list_zip)    
    return(list_dict)


# credits to https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
def save_obj(path,
             obj,
             name,
             typ = "pkl"):
    savepath = path + name + '.' + typ
    if typ == "pkl":
        with open(savepath, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    elif typ == "csv":
        with open(savepath, 'wb') as f:
            obj.to_csv(path_or_buf = savepath,
                       sep = ",",
                       index=True)


def load_obj(name,
             path,
             typ = "pkl"):
    loadpath = path + name + '.' + typ
    if typ == "pkl":
        with open(loadpath, 'rb') as f:
            return pickle.load(f)
    elif typ == "csv":
        obj = to_csv(path_or_buf = loadpath,
                     sep = ",",
                     index=True)
        return obj
    

def round_pd_series(x, freq="1d"):
    return(x.round(freq=freq))


def testplot(col,
             coindata,
             settings = SETTINGS):
    var  = coindata[col]
    time = coindata["time"]
    prices = coindata[settings["prep_and_cleaning"]["price_var"]]

    # Create some mock data
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price', color=color)
    ax1.plot(time, prices, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel(col, color=color)  # we already handled the x-label with ax1
    ax2.plot(time, var, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.ion()
    plt.show()
    
def whereami():
    whereiam = os.path.abspath(os.curdir)
    return whereiam


def display_available():
    return bool(os.environ.get('DISPLAY', None))


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__
