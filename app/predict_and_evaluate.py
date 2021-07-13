import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from plot import plot_series


def predict_VAR(fitted, test, n_in, n_out):
    y_hat_test = np.zeros((test.shape[0] - (n_in+n_out-1), n_out))
    for i in range(test.shape[0] - (n_in+n_out-1)):
        y_hat_test[i] = fitted.forecast(test.values[i:i + n_in], steps=n_out)[:, 0].reshape(-1, n_out)
    return y_hat_test


def average_prediction(y_hat, n_out):
    y_hat = np.insert(y_hat, [y_hat.shape[0]], np.empty((n_out - 1, n_out)), axis=0)
    for i in range(y_hat.shape[1]):
        y_hat[:, i] = np.roll(y_hat[:, i], i, axis=0)
    y_hat = np.nanmean(y_hat, axis=1).reshape(-1, 1)
    return y_hat


def evaluate(y_hat, n_in, n_out, test, scaler, imputed_test, scaled_test):
    if n_out > 1:
        for i in range(6):
            y_hat[i] = average_prediction(y_hat[i], n_out)
    y_hat = np.concatenate([y_hat[0], y_hat[1], y_hat[2], y_hat[3], y_hat[4], y_hat[5]], axis=1)
    y_hat = pd.concat([scaled_test.iloc[0:n_in, :6], pd.DataFrame(y_hat, columns=test.columns[:6])],
                      axis=0, ignore_index=True)
    predicted = pd.concat([y_hat, scaled_test.iloc[:, 6:]], axis=1)
    predicted = pd.DataFrame(scaler.inverse_transform(predicted), columns=test.columns, index=test.index)
    mae = []
    for i in range(6):
        mae.append(round(mean_absolute_error(imputed_test.iloc[:, i], predicted.iloc[:, i]), 3))
        # plot_series(x=test.index, y=imputed_test[:, i], z=predicted.iloc[:, i], title=test.columns[i],
        #             txt=mae, multiple_plots=True)
    return predicted, mae