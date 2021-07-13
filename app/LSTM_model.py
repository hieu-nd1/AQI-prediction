from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dataset import Dataset, preprocess_data, \
    create_data_for_each_pollutant, series_to_supervised
from tensorflow import keras
np.random.seed(7)

AQI_DATA_DIRNAME = Path(__file__).resolve().parents[0]/"dataset"/"hanoi-air-quality.csv"
WEATHER_DATA_DIRNAME = Path(__file__).resolve().parents[0]/"dataset"/"weather.csv"

data = Dataset(AQI_DATA_DIRNAME, WEATHER_DATA_DIRNAME)
column_names = data.columns

# Split train, validation and test set
train, valid, test = data.train, data.valid, data.test

# Preprocess data
_, _, _, _, _, scaled_train, scaled_valid, _ = preprocess_data(train, valid, test)

n_in = 1
n_out = 1
n_features = 5

# Split data for each pollutants
train_pm25, train_pm10, train_o3, train_no2, train_so2, train_co =\
    create_data_for_each_pollutant(scaled_train, n_features)
valid_pm25, valid_pm10, valid_o3, valid_no2, valid_so2, valid_co =\
    create_data_for_each_pollutant(scaled_valid, n_features)

train_pm25_sp = series_to_supervised(train_pm25, n_in, n_out, n_features)
train_pm10_sp = series_to_supervised(train_pm10, n_in, n_out, n_features)
train_o3_sp = series_to_supervised(train_o3, n_in, n_out, n_features)
train_no2_sp = series_to_supervised(train_no2, n_in, n_out, n_features)
train_so2_sp = series_to_supervised(train_so2, n_in, n_out, n_features)
train_co_sp = series_to_supervised(train_co, n_in, n_out, n_features)

valid_pm25_sp = series_to_supervised(valid_pm25, n_in, n_out, n_features)
valid_pm10_sp = series_to_supervised(valid_pm10, n_in, n_out, n_features)
valid_o3_sp = series_to_supervised(valid_o3, n_in, n_out, n_features)
valid_no2_sp = series_to_supervised(valid_no2, n_in, n_out, n_features)
valid_so2_sp = series_to_supervised(valid_so2, n_in, n_out, n_features)
valid_co_sp = series_to_supervised(valid_co, n_in, n_out, n_features)

X_train_pm25, y_train_pm25 = train_pm25_sp[:, np.newaxis, :-n_out].reshape((train_pm25_sp.shape[0], n_in, n_features)), \
                             train_pm25_sp[:, -n_out:]
X_valid_pm25, y_valid_pm25 = valid_pm25_sp[:, np.newaxis, :-n_out].reshape((valid_pm25_sp.shape[0], n_in, n_features)), \
                             valid_pm25_sp[:, -n_out:]
X_train_pm10, y_train_pm10 = train_pm10_sp[:, np.newaxis, :-n_out].reshape((train_pm10_sp.shape[0], n_in, n_features)), \
                             train_pm10_sp[:, -n_out:]
X_valid_pm10, y_valid_pm10 = valid_pm10_sp[:, np.newaxis, :-n_out].reshape((valid_pm10_sp.shape[0], n_in, n_features)), \
                             valid_pm10_sp[:, -n_out:]
X_train_o3, y_train_o3 = train_o3_sp[:, np.newaxis, :-n_out].reshape((train_o3_sp.shape[0], n_in, n_features)), \
                         train_o3_sp[:, -n_out:]
X_valid_o3, y_valid_o3 = valid_o3_sp[:, np.newaxis, :-n_out].reshape((valid_o3_sp.shape[0], n_in, n_features)), \
                         valid_o3_sp[:, -n_out:]
X_train_no2, y_train_no2 = train_no2_sp[:, np.newaxis, :-n_out].reshape((train_no2_sp.shape[0], n_in, n_features)), \
                           train_no2_sp[:, -n_out:]
X_valid_no2, y_valid_no2 = valid_no2_sp[:, np.newaxis, :-n_out].reshape((valid_no2_sp.shape[0], n_in, n_features)), \
                           valid_no2_sp[:, -n_out:]
X_train_so2, y_train_so2 = train_so2_sp[:, np.newaxis, :-n_out].reshape((train_so2_sp.shape[0], n_in, n_features)), \
                           train_so2_sp[:, -n_out:]
X_valid_so2, y_valid_so2 = valid_so2_sp[:, np.newaxis, :-n_out].reshape((valid_so2_sp.shape[0], n_in, n_features)), \
                           valid_so2_sp[:, -n_out:]
X_train_co, y_train_co = train_co_sp[:, np.newaxis, :-n_out].reshape((train_co_sp.shape[0], n_in, n_features)), \
                         train_co_sp[:, -n_out:]
X_valid_co, y_valid_co = valid_co_sp[:, np.newaxis, :-n_out].reshape((valid_co_sp.shape[0], n_in, n_features)), \
                         valid_co_sp[:, -n_out:]

# Create LSTM model
model = keras.models.Sequential([
    keras.layers.LSTM(10, input_shape=[n_in, n_features]),
    keras.layers.Dense(n_out)
])


# Train LSTM model and predict
def training_LSTM(model, X_train , y_train, X_valid, y_valid, pos):
    model.compile(loss="mae", optimizer='adam')
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_valid, y_valid))
    model.save(f'weights_LSTM/{n_in}in-{n_out}out/{column_names[pos]}.h5')
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='valid')
    # plt.legend()
    # plt.show()


training_LSTM(model, X_train_pm25, y_train_pm25, X_valid_pm25, y_valid_pm25, 0)
training_LSTM(model, X_train_pm10, y_train_pm10, X_valid_pm10, y_valid_pm10, 1)
training_LSTM(model, X_train_o3, y_train_o3, X_valid_o3, y_valid_o3, 2)
training_LSTM(model, X_train_no2, y_train_no2, X_valid_no2, y_valid_no2, 3)
training_LSTM(model, X_train_so2, y_train_so2, X_valid_so2, y_valid_so2, 4)
training_LSTM(model, X_train_co, y_train_co, X_valid_co, y_valid_co, 5)