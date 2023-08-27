import numpy as np
from scipy.special import comb

def _zscore_for_pd(pd_col,
                   stdev_degrees_of_free):
    zscores = (pd_col - pd_col.mean())/pd_col.std(ddof=stdev_degrees_of_free)
    return zscores


def _norm_for_pd(pd_col):
    norm = (pd_col - pd_col.min())/(pd_col.max()-pd_col.min())
    return norm


def truncate_outliers(df,
                      settings,
                      cols="from_config",
                      threshold = "from_config"):
        # Input either from config file or specified in arguments
    if threshold == "from_config":
        threshold = settings["prep_and_cleaning"]["outlier_threshold_for_zscore"]
    elif not type(threshold) == int:
        raise TypeError("Input has to be an integer.")

    if cols == "from_config":
        cols = settings["prep_and_cleaning"]["outlier_cols_to_be_treated"]

    for col in cols:
        threshval_neg = (df[col].mean() - df[col].std(ddof=1))*threshold
        threshval_pos = (df[col].mean() + df[col].std(ddof=1))*threshold
        zscores   = _zscore_for_pd(pd_col = df[col],
                                   stdev_degrees_of_free = 1) # Bessels correction for sample stdev
        is_outl_neg   = zscores < -threshold
        is_outl_pos   = zscores > threshold
        df.loc[is_outl_neg, [col]] = threshval_neg
        df.loc[is_outl_pos, [col]] = threshval_pos

        print("|---| Truncated outliers for column: " + str(col) + " |---| ")
        return df


def normalize_data_by_col(df,
                          settings,
                          cols   = "from_config",
                          typ    = "from_config"):
    # Input either from config file or specified in arguments
    if typ == "from_config":
        typ = settings["prep_and_cleaning"]["normalization_type"]
    elif not typ in ["normalization", "standardization"]:
        raise TypeError("Input has to be either 'standardization' or 'normalization'.")

    if cols == "from_config":
        cols = settings["prep_and_cleaning"]["normalization_columns"]

    for col in cols:
        coldta      = df[col]
        if(typ == "standardization"):
            zscores   = _zscore_for_pd(pd_col = coldta,
                                       stdev_degrees_of_free = 1) # Bessels correction for sample stdev
            df.loc[:, [col]] = zscores
        elif(typ == "normalization"):
            normalized   = _norm_for_pd(pd_col = coldta) # Bessels correction for sample stdev
            df.loc[:, [col]] = normalized

    print("|---| Normalized data |---| ")


def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


def smoothstep(x, x_min=0, x_max=1, N=1):
    # https://stackoverflow.com/questions/45165452/how-to-implement-a-smooth-clamp-function-in-python
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


def smooth_steps_left_cetered(vec, N):
    result = vec.rolling(window=N).mean().shift(-N)
    return result
