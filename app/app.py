from pathlib import Path
import pandas as pd
import numpy as np
import os
from dataset import Dataset, preprocess_data, \
    create_data_for_each_pollutant, series_to_supervised
from plot import plot_series, plot_percent_missing, plot_histogram
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from predict_and_evaluate import predict_VAR, evaluate
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from statsmodels.api import load_pickle

AQI_DATA_DIRNAME = Path(__file__).resolve().parents[0]/"dataset"/"hanoi-air-quality.csv"
WEATHER_DATA_DIRNAME = Path(__file__).resolve().parents[0]/"dataset"/"weather.csv"

data = Dataset(AQI_DATA_DIRNAME, WEATHER_DATA_DIRNAME)
column_names = data.columns

# Plot time series before preprocessing
# plot_series(x=data.index, y=data.pm25, title="pm25")
# plot_series(x=data.index, y=data.pm10, title="pm10")
# plot_series(x=data.index, y=data.o3, title="o3")
# plot_series(x=data.index, y=data.no2, title="no2")
# plot_series(x=data.index, y=data.so2, title="so2")
# plot_series(x=data.index, y=data.co, title="co")

# plot_percent_missing(data)
# plot_histogram(data)

# Split train, validation and test set
train, valid, test = data.train, data.valid, data.test

# Preprocess data
_, scaler, imputed_train, imputed_valid, imputed_test, _, _, scaled_test = preprocess_data(train, valid, test)

# Plot time series after preprocessing
imputed_data = pd.DataFrame(np.concatenate([imputed_train, imputed_valid, imputed_test], axis=0), columns=data.columns)
imputed_test = pd.DataFrame(imputed_test, columns=test.columns, index=test.index)

# plot_series(x=data.index, y=imputed_data.pm25, title='pm25')
# plot_series(x=data.index, y=imputed_data.pm10, title="pm10")
# plot_series(x=data.index, y=imputed_data.o3, title="o3")
# plot_series(x=data.index, y=imputed_data.no2, title="no2")
# plot_series(x=data.index, y=imputed_data.so2, title="so2")
# plot_series(x=data.index, y=imputed_data.co, title="co")
# plot_histogram(imputed_data)

n_features = 5
# Split data for each pollutants
test_pm25, test_pm10, test_o3, test_no2, test_so2, test_co = \
    create_data_for_each_pollutant(scaled_test, n_features)

app = dash.Dash(__name__, external_stylesheets=[])
app.title = "Air Quality Prediction"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(children="Air Quality Prediction", className="header-title"),
                html.P(
                    children="Analysis and prediction of air quality data"
                             " between 2016 and 2020",
                    className="header-description"
                )
            ],
            className="header"
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Pollutant", className="menu-title"),
                        dcc.Dropdown(
                            id="pollutant",
                            options=[
                                {"label": column_name, "value": column_name}
                                for column_name in column_names[:6]
                            ],
                            value=column_names[0],
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Lag Order", className="menu-title"),
                        dcc.Dropdown(
                            id="n_in",
                            options=[
                                {"label": i+1, "value": i+1}
                                for i in range(7)
                            ],
                            value=1,
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Prediction steps", className="menu-title"),
                        dcc.Dropdown(
                            id="n_out",
                            options=[
                                {"label": i+1, "value": i+1}
                                for i in range(2)
                            ],
                            value=1,
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range",
                            className="menu-title"
                        ),
                        dcc.DatePickerRange(
                            id="date_range",
                            min_date_allowed=test.index.min().date(),
                            max_date_allowed=test.index.max().date(),
                            start_date=test.index.min().date(),
                            end_date=test.index.max().date(),
                            display_format='D-M-Y'
                        ),
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                dcc.Loading(
                    id="loading",
                    type="default",
                    children=[
                        html.Div(
                            children=dcc.Graph(
                                id="AQI-VAR", config={"displayModeBar": False},
                            ),
                            className="card",
                        ),
                        html.Div(
                            children=dcc.Graph(
                                id="AQI-LSTM", config={"displayModeBar": False},
                            ),
                            className="card",
                        ),
                    ],
                ),
            ],
            className="wrapper",
        ),
    ]
)


@app.callback(
    [Output("AQI-VAR", "figure"), Output("AQI-LSTM", "figure")],
    [
        Input("pollutant", "value"),
        Input("n_in", "value"),
        Input("n_out", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
    ],
)
def update_charts(pollutant, n_in, n_out, start_date, end_date):
    fitted_pm25 = load_pickle(f'weights_VAR/{n_in}in-{n_out}out/{column_names[0]}.pkl')
    fitted_pm10 = load_pickle(f'weights_VAR/{n_in}in-{n_out}out/{column_names[1]}.pkl')
    fitted_o3 = load_pickle(f'weights_VAR/{n_in}in-{n_out}out/{column_names[2]}.pkl')
    fitted_no2 = load_pickle(f'weights_VAR/{n_in}in-{n_out}out/{column_names[3]}.pkl')
    fitted_so2 = load_pickle(f'weights_VAR/{n_in}in-{n_out}out/{column_names[4]}.pkl')
    fitted_co = load_pickle(f'weights_VAR/{n_in}in-{n_out}out/{column_names[5]}.pkl')

    y_hat_pm25_1 = predict_VAR(fitted_pm25, test_pm25, n_in, n_out)
    y_hat_pm10_1 = predict_VAR(fitted_pm10, test_pm10, n_in, n_out)
    y_hat_o3_1 = predict_VAR(fitted_o3, test_o3.iloc[:, :4], n_in, n_out)
    y_hat_no2_1 = predict_VAR(fitted_no2, test_no2, n_in, n_out)
    y_hat_so2_1 = predict_VAR(fitted_so2, test_so2, n_in, n_out)
    y_hat_co_1 = predict_VAR(fitted_co, test_co, n_in, n_out)

    predicted_VAR, mae_VAR = evaluate([y_hat_pm25_1, y_hat_pm10_1, y_hat_o3_1, y_hat_no2_1, y_hat_so2_1, y_hat_co_1],
                                      n_in, n_out, test, scaler, imputed_test, scaled_test)

    test_pm25_sp = series_to_supervised(test_pm25, n_in, n_out, n_features)
    test_pm10_sp = series_to_supervised(test_pm10, n_in, n_out, n_features)
    test_o3_sp = series_to_supervised(test_o3, n_in, n_out, n_features)
    test_no2_sp = series_to_supervised(test_no2, n_in, n_out, n_features)
    test_so2_sp = series_to_supervised(test_so2, n_in, n_out, n_features)
    test_co_sp = series_to_supervised(test_co, n_in, n_out, n_features)

    X_test_pm25 = test_pm25_sp[:, np.newaxis, :-n_out].reshape((test_pm25_sp.shape[0], n_in, n_features))
    X_test_pm10 = test_pm10_sp[:, np.newaxis, :-n_out].reshape((test_pm10_sp.shape[0], n_in, n_features))
    X_test_o3 = test_o3_sp[:, np.newaxis, :-n_out].reshape((test_o3_sp.shape[0], n_in, n_features))
    X_test_no2 = test_no2_sp[:, np.newaxis, :-n_out].reshape((test_no2_sp.shape[0], n_in, n_features))
    X_test_so2 = test_so2_sp[:, np.newaxis, :-n_out].reshape((test_so2_sp.shape[0], n_in, n_features))
    X_test_co = test_co_sp[:, np.newaxis, :-n_out].reshape((test_co_sp.shape[0], n_in, n_features))

    model = keras.models.load_model(f'weights_LSTM/{n_in}in-{n_out}out/{column_names[0]}.h5')
    y_hat_pm25_2 = model.predict(X_test_pm25)
    model = keras.models.load_model(f'weights_LSTM/{n_in}in-{n_out}out/{column_names[1]}.h5')
    y_hat_pm10_2 = model.predict(X_test_pm10)
    model = keras.models.load_model(f'weights_LSTM/{n_in}in-{n_out}out/{column_names[2]}.h5')
    y_hat_o3_2 = model.predict(X_test_o3)
    model = keras.models.load_model(f'weights_LSTM/{n_in}in-{n_out}out/{column_names[3]}.h5')
    y_hat_no2_2 = model.predict(X_test_no2)
    model = keras.models.load_model(f'weights_LSTM/{n_in}in-{n_out}out/{column_names[4]}.h5')
    y_hat_so2_2 = model.predict(X_test_so2)
    model = keras.models.load_model(f'weights_LSTM/{n_in}in-{n_out}out/{column_names[5]}.h5')
    y_hat_co_2 = model.predict(X_test_co)

    predicted_LSTM, mae_LSTM = evaluate([y_hat_pm25_2, y_hat_pm10_2, y_hat_o3_2, y_hat_no2_2, y_hat_so2_2,
                                         y_hat_co_2], n_in, n_out, test, scaler, imputed_test, scaled_test)

    filtered_test = imputed_test.loc[start_date:end_date, pollutant]
    filtered_VAR = predicted_VAR.loc[start_date:end_date, pollutant]
    filtered_LSTM = predicted_LSTM.loc[start_date:end_date, pollutant]
    VAR_figure = {
        "data": [
            {
                "x": filtered_test.index,
                "y": filtered_test,
                "type": "lines",
                "name": "Actual",
            },
            {
                "x": filtered_VAR.index,
                "y": filtered_VAR,
                "type": "lines",
                "name": "Predicted",
            },
        ],
        "layout": {
            "title": {
                "text": "AQI prediction by VAR model",
                "x": 0.39,
                "xanchor": "left",
            },
            "showlegend": True,
            "legend": {
                "x": 0,
                "y": 1
            },
            "annotations": [{
                "text": f"Loss: {mae_VAR[imputed_test.columns.get_loc(pollutant)]}",
                "x": 0.5, "y": 1, "yanchor": "bottom", "font": {"family": "sans-serif", "size": 18},
                "xref": "paper", "yref": "paper","showarrow": False,
            }],
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
        },
    }

    LSTM_figure = {
        "data": [
            {
                "x": filtered_test.index,
                "y": filtered_test,
                "type": "lines",
                "name": "Actual",
            },
            {
                "x": filtered_LSTM.index,
                "y": filtered_LSTM,
                "type": "lines",
                "name": "Predicted",
            },
        ],
        "layout": {
            "title": {
                "text": "AQI prediction by LSTM model",
                "x": 0.39,
                "xanchor": "left",
            },
            "showlegend": True,
            "legend": {
                "x": 0,
                "y": 1
            },
            "annotations": [{
                "text": f"Loss: {mae_LSTM[imputed_test.columns.get_loc(pollutant)]}",
                "x": 0.5, "y": 1, "yanchor": "bottom", "font": {"family": "sans-serif", "size": 18},
                "xref": "paper", "yref": "paper","showarrow": False,
            }],
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
        },
    }
    return VAR_figure, LSTM_figure


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
