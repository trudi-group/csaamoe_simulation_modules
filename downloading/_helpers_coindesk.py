import requests
import json
import pandas as pd

def get_coindesk_data(day_start
                      , day_end):
    # From: https://www.coindesk.com/coindesk-api
    # Download data as JSON
    url_coindesk = "https://api.coindesk.com/v1/bpi/historical/close.json?" \
                   "start={}&end={}".format(day_start, day_end)
    r = requests.get(url=url_coindesk)
    dta_raw = json.loads(r.content.decode())
    # Put data into Pandas
    dta = pd.DataFrame.from_dict(dta_raw["bpi"], orient="index")
    # Standardize data: Sort, set column-names, etc.
    dta = dta.sort_index()
    dta.columns = ["bpi_usd"]
    dta["time"] = dta.index

    return dta
