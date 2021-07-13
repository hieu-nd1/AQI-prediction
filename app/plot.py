import matplotlib.pyplot as plt
import numpy as np


# Plot time series of 6 pollutants
def plot_series(x, y, z=None, title="", xlabel='Date', ylabel='Value', txt=None, multiple_plots=False):
    plt.figure(figsize=(16,5))
    plt.plot(x, y, label='real')
    if multiple_plots:
        plt.plot(x, z, label='predicted')
        plt.annotate(f'Loss: {txt}', xy=(0.45, 0.9), xycoords='axes fraction')
        plt.legend()
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# Plot % of missing data
def plot_percent_missing(data):
    def percent_missing(p):
        return p.isnull().sum() * 100 / len(p)
    fig, ax = plt.subplots()
    pollutants = ('pm25', 'pm10', 'o3', 'no2', 'so2', 'co')
    y = np.arange(len(pollutants))
    p = np.round([percent_missing(data.pm25), percent_missing(data.pm10), percent_missing(data.o3),
                  percent_missing(data.no2), percent_missing(data.so2), percent_missing(data.co)], 1)
    rect = ax.bar(y, p, 0.35)
    plt.xticks(y, pollutants)
    plt.ylabel('Percentage')
    plt.title('Percentage of missing values in Six Pollutants')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rect)
    plt.show()


# Plot Histogram
def plot_histogram(data):
    plt.figure(figsize=(16,5))
    for i in range(1,7):
        plt.subplot(2,3,i)
        data.iloc[:,i].hist()
        plt.title(data.columns[i-1])
        plt.tight_layout()
    plt.show()