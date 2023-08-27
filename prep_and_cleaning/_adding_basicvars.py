
def add_returns(df
                , col_price):
        prices = df[col_price]
        returns = prices / prices.shift(-1) - 1
        df["returns"] = returns

        print("|---| Added returns |---| ")
        return df


def add_volatility_squared_returns(df
                                  , col_returns):
    returns = df[col_returns]
    volatility = returns.pow(2)
    df["volatility_sr"] = volatility

    print("|---| Added volatility (squared returns) |---| ")


def add_shifted_var(df
                    , col_to_shift
                    , typ):
    if typ == "lead":
        shift_parameter = +1
    elif typ == "lag":
        shift_parameter = -1
    else:
        raise Exception("Wrong input!")

    df[col_to_shift+"_"+typ] = df[col_to_shift].shift(shift_parameter)
    print("|---| Added returns |---|")
    return df


def add_velocity(df
    , col_tx_vol
    , col_m_circ
    , name_suffix=""):

    df["velocity"+name_suffix] = df[col_tx_vol]/df[col_m_circ]

    return df
