from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np


# Check if time series is stationary by using ADFuller Test
def adfuller_test(data, signif=0.05, name=''):
    result = adfuller(data, autolag="AIC")
    statistic = round(result[0], 3)
    p_value = round(result[1], 3)

    print(f' Augmented Dickey-Fuller Test on "{name}":')
    print(f' Significance Level = {signif}')
    print(f' ADF Statistic     = {statistic}')
    for key, value in result[4].items():
        print(f' Critical value {key} = {round(value, 3)}')
    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


# Check Granger’s Causality
def grangers_causation_matrix(data, aqi, weather, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((6, 4)), columns=weather, index=aqi)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=12, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(12)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in weather]
    df.index = [var + '_y' for var in aqi]
    return df