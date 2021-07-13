import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

AQI_DATA_DIRNAME = Path(__file__).resolve().parents[0]/"dataset"/"hanoi-air-quality.csv"
WEATHER_DATA_DIRNAME = Path(__file__).resolve().parents[0]/"dataset"/"weather.csv"


class Dataset(pd.DataFrame):
    def __init__(self, aqi_file, weather_file):
        super().__init__(merge_datasets(aqi_file, weather_file))

    @property
    def train(self):
        return self.query('date <= "2018-12-31"')

    @property
    def valid(self):
        return self.query('date >= "2019-01-01" and date <= "2019-12-31"')

    @property
    def test(self):
        return self.query('date >= "2020-01-01"')


def merge_datasets(aqi_file, weather_file):
    # Get the AQI data
    aqi_data = pd.read_csv(aqi_file, parse_dates=["date"], date_parser=pd.to_datetime)
    aqi_data.set_index(keys=["date"], inplace=True)
    dates = list(pd.date_range(min(aqi_data.index), max(aqi_data.index), freq='D').values)
    aqi_data = aqi_data.reindex(dates)

    # Get weather data and merge with AQI data
    weather_data = pd.read_csv(weather_file, parse_dates=["date"], date_parser=pd.to_datetime)
    weather_data.set_index(keys=["date"], inplace=True)

    data = pd.merge(aqi_data, weather_data, left_index=True, right_index=True)
    return data


def preprocess_data(train, valid, test):
    imputer = KNNImputer(n_neighbors=3)
    imputed_train = imputer.fit_transform(train)
    imputed_valid = imputer.transform(valid)
    imputed_test = imputer.transform(test)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = pd.DataFrame(scaler.fit_transform(imputed_train), columns=train.columns)
    scaled_valid = pd.DataFrame(scaler.transform(imputed_valid), columns=valid.columns)
    scaled_test = pd.DataFrame(scaler.fit_transform(imputed_test), columns=test.columns)
    return imputer, scaler,\
           imputed_train, imputed_valid, imputed_test,\
           scaled_train, scaled_valid, scaled_test


def create_data_for_each_pollutant(data, n_features):
    splitdata = []
    for i in range(6):
        if n_features == 1:
            splitdata.append(data.iloc[:, [i]])
        if n_features == 2:
            splitdata.append(data.iloc[:, [i, 6]])
        if n_features == 3:
            splitdata.append(data.iloc[:, [i, 6, 7]])
        if n_features == 4:
            splitdata.append(data.iloc[:, [i, 6, 7, 8]])
        if n_features == 5:
            splitdata.append(data.iloc[:, [i, 6, 7, 8, 9]])
    return splitdata[0], splitdata[1], splitdata[2], splitdata[3], splitdata[4], splitdata[5]


def series_to_supervised(data, n_in, n_out, n_features, dropnan=True):
    n_vars = data.shape[1]
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (data.columns[j], i)) for j in range(n_vars)]
    for i in range(n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [('%s(t)' % (data.columns[j])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (data.columns[j], i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    if n_features > 1:
        agg.drop(agg.columns[-n_features+1:], axis=1, inplace=True)
        if n_out > 1:
            for i in range(1, n_out):
                agg.drop(agg.columns[-n_features-(i-1):-i], axis=1, inplace=True)
    return agg.values