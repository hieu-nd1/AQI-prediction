from pathlib import Path
import numpy as np
from dataset import Dataset, preprocess_data, \
    create_data_for_each_pollutant
from statistical_tests import adfuller_test, grangers_causation_matrix
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error
from predict_and_evaluate import average_prediction

AQI_DATA_DIRNAME = Path(__file__).resolve().parents[0]/"dataset"/"hanoi-air-quality.csv"
WEATHER_DATA_DIRNAME = Path(__file__).resolve().parents[0]/"dataset"/"weather.csv"

data = Dataset(AQI_DATA_DIRNAME, WEATHER_DATA_DIRNAME)
column_names = data.columns

train, valid, test = data.train, data.valid, data.test

_, _, imputed_train, imputed_valid, imputed_test, scaled_train, scaled_valid, _ = preprocess_data(train, valid, test)

imputed_data = pd.DataFrame(np.concatenate([imputed_train, imputed_valid, imputed_test], axis=0), columns=data.columns)

# for name, column in imputed_data.iloc[:, 0:6].iteritems():
#     adfuller_test(column, name=column.name)
#     print('\n')

# print(grangers_causation_matrix(imputed_data, weather=column_names[6:], aqi=column_names[:6]))

n_in = 1
n_out = 1
n_features = 5

# Split data for each pollutants
train_pm25, train_pm10, train_o3, train_no2, train_so2, train_co =\
    create_data_for_each_pollutant(scaled_train, n_features)
valid_pm25, valid_pm10, valid_o3, valid_no2, valid_so2, valid_co =\
    create_data_for_each_pollutant(scaled_valid, n_features)


# Train Vector Autoregression model
def fit_VAR(train, valid, n_in, n_out, pos):
    model = VAR(train)
    # print(model.select_order())
    fitted_model = model.fit(n_in)
    # print_loss(train, valid, fitted_model, n_in, n_out)
    fitted_model.save(f'weights_VAR/{n_in}in-{n_out}out/{column_names[pos]}.pkl')
    return fitted_model


def print_loss(train, valid, fitted_model, n_in, n_out):
    y_hat_train = np.zeros((train.shape[0] - (n_in+n_out-1), n_out))
    for i in range(train.shape[0] - (n_in+n_out-1)):
        y_hat_train[i] = fitted_model.forecast(train.values[i:i + n_in], steps=n_out)[:, 0].reshape(-1, n_out)
    y_hat_train = average_prediction(y_hat_train, n_out)
    y_hat_valid = np.zeros((valid.shape[0] - (n_in+n_out-1), n_out))
    for i in range(valid.shape[0] - (n_in+n_out-1)):
        y_hat_valid[i] = fitted_model.forecast(valid.values[i:i + n_in], steps=n_out)[:, 0].reshape(-1, n_out)
    y_hat_valid = average_prediction(y_hat_valid, n_out)
    print('loss: %f' %(mean_absolute_error(train.values[n_in:, 0], y_hat_train)),
          'val_loss: %f' %mean_absolute_error(valid.values[n_in:, 0], y_hat_valid))


fitted_pm25 = fit_VAR(train_pm25, valid_pm25, n_in, n_out, 0)
fitted_pm10 = fit_VAR(train_pm10, valid_pm10, n_in, n_out, 1)
fitted_o3 = fit_VAR(train_o3.iloc[:, :4], valid_o3.iloc[:, :4], n_in, n_out, 2)
fitted_no2 = fit_VAR(train_no2, valid_no2, n_in, n_out, 3)
fitted_so2 = fit_VAR(train_so2, valid_so2, n_in, n_out, 4)
fitted_co = fit_VAR(train_co, valid_co, n_in, n_out, 5)

