import quandl


def get_quandl_data(day_start
                    , day_end
                    , SETTINGS):
    quandl.ApiConfig.api_key = open(SETTINGS["downloads"]["quandl"]["api_key_path"], "r").read().replace("\n", "")
    ids = SETTINGS["downloads"]["quandl"]["ids"].split(", ")
    dta = quandl.get(ids
                     , paginate=True
                     , start_date=day_start
                     , end_date=day_end)
    dta.columns = ids
    dta["time"] = dta.index.strftime("%Y-%m-%d")

    return dta

