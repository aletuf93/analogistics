# %% import packages

import numpy as np
import pandas as pd
import itertools
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

from pandas.api.types import CategoricalDtype

from scipy.stats import boxcox


def timeStampToDays(series: pd.Series) -> pd.Series:
    """
    Convert a datetime series into float series with the number of days
    :param series: DESCRIPTION input pandas series
    :type series: pd.Series
    :return: DESCRIPTION pandas series with float of days
    :rtype: TYPE pd.Series

    """

    D = series.dt.components['days']
    H = series.dt.components['hours']
    M = series.dt.components['minutes']
    result = D + (H / 24) + (M / (60 * 24))
    return result


def sampleTimeSeries(series: pd.Series,
                     sampleInterval: str) -> pd.Series:
    """
    Sample a pandas series using a sampling interval
    :param series: DESCRIPTION input pandas datetime series
    :type series: pd.Series
    :param sampleInterval: DESCRIPTION type of sampling required
    :type sampleInterval: str
    :raises ValueError: DESCRIPTION error in case of invalid sampling parameter
    :return: DESCRIPTION Output sampled seried
    :rtype: TYPE pd.Series

    """

    if sampleInterval not in ['day', 'week', 'month', 'year']:
        raise ValueError(f"""sampleInterval parameter: {sampleInterval} not a valid sample interval.
                          Choose between ['day', 'week', 'month', 'year']""")
    if sampleInterval == 'day':
        series = series.dt.strftime('%Y-%j')
    elif sampleInterval == 'week':
        series = series.dt.strftime('%Y-%U')
    elif sampleInterval == 'month':
        series = series.dt.strftime('%Y-%m')
    elif sampleInterval == 'year':
        series = series.dt.strftime('%Y')
    return series


def groupPerWeek(df: pd.DataFrame,
                 timeVariable: str,
                 groupVariable: str,
                 groupType: str) -> pd.DataFrame:
    """
    Perform a weekly groupby based on a datetime variable, applying a specific type of grouping
    :param df: DESCRIPTION input pandas dataframe
    :type df: pd.DataFrame
    :param timeVariable: DESCRIPTION column name corresponding to the time variable
    :type timeVariable: str
    :param groupVariable: DESCRIPTION column name corresponding to the grouping variable
    :type groupVariable: str
    :param groupType: DESCRIPTION type of grouping function
    :type groupType: str
    :return: DESCRIPTION Output grouped DataFrame
    :rtype: TYPE pd.DataFrame

    """

    if groupType not in ['count', 'sum']:
        raise ValueError(f"""groupType parameter: {groupType} not a valid grouping function.
                          Choose between ['count', 'sum']""")

    # convert to dataframe if a series
    if isinstance(df, pd.Series):
        df = pd.DataFrame([[df.index.values.T, df.values]],
                          columns=[timeVariable, groupVariable])

    df['DatePeriod'] = pd.to_datetime(df[timeVariable]) - pd.to_timedelta(7, unit='d')

    if groupType == 'count':
        df = df.groupby([pd.Grouper(key=timeVariable,
                                    freq='W-MON')])[groupVariable].size()
    elif groupType == 'sum':
        df = df.groupby([pd.Grouper(key=timeVariable,
                                    freq='W-MON')])[groupVariable].sum()
    df = df.sort_index()
    return df


def groupPerMonth(df: pd.DataFrame,
                  timeVariable: str,
                  groupVariable: str,
                  groupType: str) -> pd.DataFrame:
    """
    Perform a monthly groupby based on a datetime variable, applying a specific type of grouping
    :param df: DESCRIPTION input pandas dataframe
    :type df: pd.DataFrame
    :param timeVariable: DESCRIPTION column name corresponding to the time variable
    :type timeVariable: str
    :param groupVariable: DESCRIPTION column name corresponding to the grouping variable
    :type groupVariable: str
    :param groupType: DESCRIPTION type of grouping function
    :type groupType: str
    :return: DESCRIPTION Output grouped DataFrame
    :rtype: TYPE pd.DataFrame

    """

    if groupType not in ['count', 'sum']:
        raise ValueError(f"""groupType parameter: {groupType} not a valid grouping function.
                          Choose between ['count', 'sum']""")

    if isinstance(df, pd.Series):  # convert to dataframe if a series
        df = pd.DataFrame([[df.index.values.T, df.values]],
                          columns=[timeVariable, groupVariable])

    # df['DatePeriod'] = pd.to_datetime(df[timeVariable]) - pd.to_timedelta(7, unit='d')

    if groupType == 'count':
        df = df.groupby([pd.Grouper(key=timeVariable, freq='M')])[groupVariable].size()
    elif groupType == 'sum':
        df = df.groupby([pd.Grouper(key=timeVariable, freq='M')])[groupVariable].sum()
    df = df.sort_index()
    return df


def groupPerWeekday(df: pd.DataFrame,
                    timeVariable: str,
                    groupVariable: str) -> pd.DataFrame:
    """
    Perform a groupby per weekday based on a datetime variable, applying a specific type of grouping
    :param df: DESCRIPTION input pandas dataframe
    :type df: pd.DataFrame
    :param timeVariable: DESCRIPTION column name corresponding to the time variable
    :type timeVariable: str
    :param groupVariable: DESCRIPTION column name corresponding to the grouping variable
    :type groupVariable: str
    :return: DESCRIPTION Output grouped DataFrame
    :rtype: TYPE pd.DataFrame

    """

    cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    cat_type = CategoricalDtype(categories=cats, ordered=True)

    df['Weekday'] = df[timeVariable].dt.day_name()
    df['Weekday'] = df['Weekday'].astype(cat_type)
    D_grouped = df.groupby(['Weekday']).agg({groupVariable: ['size', 'mean', 'std']})
    D_grouped.columns = D_grouped.columns.droplevel(0)
    D_grouped['mean'] = np.round(D_grouped['mean'], 2)
    D_grouped['std'] = np.round(D_grouped['std'], 2)
    return D_grouped


def assignWeekDay(df: pd.DataFrame,
                  timeVariable: str) -> tuple:
    """
    Return the day of the week, and a boolean indicating whether the day is in the weekend
    :param df: DESCRIPTION input pandas dataframe
    :type df: pd.DataFrame
    :param timeVariable: DESCRIPTION column name corresponding to the time variable
    :type timeVariable: str
    :return: DESCRIPTION tuple with the day of the week, and a boolean for weekend
    :rtype: tuple

    """
    dayOfTheWeek = df[timeVariable].dt.weekday_name
    weekend = (dayOfTheWeek == 'Sunday') | (dayOfTheWeek == 'Saturday')
    weekEnd = weekend.copy()
    weekEnd[weekend] = 'Weekend'
    weekEnd[~weekend] = 'Weekday'
    return dayOfTheWeek, weekEnd


def ACF_PACF_plot(series: pd.Series) -> tuple:
    """
    Creates a graph with a time series, the ACF and the PACF. In addition, it returns
    two pandas Series with the significant lags in the ACF and PACF
    :param series: DESCRIPTION input pandas series with the observations
    :type series: pd.Series
    :return: DESCRIPTION output tuple
    :rtype: tuple

    """

    # Prepare the output figure
    fig = plt.subplot(131)

    plt.plot(series, 'skyblue')
    plt.xticks(rotation=30)
    plt.title('Time Series')

    lag_acf = acf(series, nlags=20)
    lag_pacf = pacf(series, nlags=20)

    plt.subplot(132)
    plt.stem(lag_acf, linefmt='skyblue', markerfmt='d')
    plt.axhline(y=0, linestyle='--')
    plt.axhline(y=-1.96 / np.sqrt(len(series)), linestyle='--', color='r')
    plt.axhline(y=1.96 / np.sqrt(len(series)), linestyle='--', color='r')
    plt.title('ACF')
    plt.xlabel('time lag')
    plt.ylabel('ACF value')

    plt.subplot(133)
    plt.stem(lag_pacf, linefmt='skyblue', markerfmt='d')
    plt.axhline(y=0, linestyle='--')
    plt.axhline(y=-1.96 / np.sqrt(len(series)), linestyle='--', color='r')
    plt.axhline(y=1.96 / np.sqrt(len(series)), linestyle='--', color='r')
    plt.title('PACF')
    plt.xlabel('time lag')
    plt.ylabel('PACF value')

    # identify significant values for ACF
    D_acf = pd.DataFrame(lag_acf, columns=['ACF'])
    D_acf['ORDER'] = D_acf.index.values + 1

    min_sign = -1.96 / np.sqrt(len(series))
    max_sign = 1.96 / np.sqrt(len(series))

    D_acf['SIGNIFICANT'] = (D_acf['ACF'] > max_sign) | (D_acf['ACF'] < min_sign)
    D_acf_significant = D_acf['ORDER'][D_acf['SIGNIFICANT']].values

    # identify significant values for PACF
    D_pacf = pd.DataFrame(lag_pacf, columns=['PACF'])
    D_pacf['ORDER'] = D_pacf.index.values + 1

    D_pacf['SIGNIFICANT'] = (D_pacf['PACF'] > max_sign) | (D_pacf['PACF'] < min_sign)
    D_pacf_significant = D_pacf['ORDER'][D_pacf['SIGNIFICANT']].values

    return fig, D_acf_significant, D_pacf_significant


def returnSignificantLags(D_pacf_significant: pd.Series,
                          D_acf_significant: pd.Series,
                          maxValuesSelected: int = 2) -> list:
    """
    This function returns tuples of significant order (p, d, q) based on the lags of the function ACF_PACF_plot

    :param D_pacf_significant: DESCRIPTION significant lags of the PACF function, like in the output of ACF_PACF_plot function
    :type D_pacf_significant: pd.Series
    :param D_acf_significant: DESCRIPTION significant lags of the ACF function, like in the output of ACF_PACF_plot function
    :type D_acf_significant: pd.Series
    :param maxValuesSelected: DESCRIPTION, defaults to 2. Number of combinations of p, d, and q to produce
    :type maxValuesSelected: int, optional
    :return: DESCRIPTION multidimensional list with combinations of (p, d, q) for ARIMA fitting
    :rtype: list

    """
    # Select values for parameter p
    if len(D_pacf_significant) > 1:
        numSelected = min(maxValuesSelected, len(D_pacf_significant))
        p = D_pacf_significant[0: numSelected]

    else:
        p = [0, 1]

    # Select values for parameter q
    if len(D_acf_significant) > 1:
        numSelected = min(maxValuesSelected, len(D_acf_significant))
        q = D_acf_significant[0: numSelected]
    else:
        q = [0, 1]

    d = [0, 1]
    a = [p, d, q]
    params = list(itertools.product(*a))
    return params


def detrendByRollingMean(series: pd.Series,
                         seasonalityPeriod: int) -> pd.Series:
    """
    Apply detrending by using a rolling mean
    :param series: DESCRIPTION input pandas series
    :type series: pd.Series
    :param seasonalityPeriod: DESCRIPTION window of the rolling mean
    :type seasonalityPeriod: int
    :return: DESCRIPTION output detrended series
    :rtype: TYPE

    """
    rolling_mean = series.rolling(window=seasonalityPeriod).mean()
    detrended = series.Series - rolling_mean
    return detrended


def SARIMAXfit(stationary_series: pd.Series,
               params: list) -> tuple:
    """
    this function tries different SARIMAX fits using tuples of orders specified in the list of tuples (p,d,q) param
    on the time series stationary_series
    the function return a figure_forecast with the plot of the forecast
    a figure_residuals with the plot of the residuals
    a dict resultModel with the model, the error (AIC), the order p,d,q

    PACF=>AR
    ACF=>MA
    ARIMA(P,D,Q) = ARIMA(AR, I, MA)

    :param stationary_series: DESCRIPTION input pandas series to fit
    :type stationary_series: pd.series
    :param params: DESCRIPTION (p, d, q) parameters to fit the SARIMAX model, as output of returnSignificantLags function
    :type params: list
    :return: DESCRIPTION tuple with output
    :rtype: tuple

    """

    # Set an initial dummy error
    incumbentError = 999999999999999999999
    bestModel = []

    for param in params:
        mod = sm.tsa.statespace.SARIMAX(stationary_series,
                                        order=param,
                                        enforce_stationarity=True,
                                        enforce_invertibility=True,
                                        initialization='approximate_diffuse')

        results = mod.fit()
        if(results.aic < incumbentError):
            bestModel = mod
            incumbentError = results.aic

    # save the best fit model
    results = bestModel.fit()
    figure_residuals = results.plot_diagnostics(figsize=(15, 12))

    # Produce output figure
    figure_forecast = plt.figure()
    plt.plot(stationary_series)
    plt.plot(results.fittedvalues, color='red')
    plt.title('ARIMA fit p=' + str(bestModel.k_ar) + ' d=' + str(bestModel.k_diff) + ' q=' + str(bestModel.k_ma))

    resultModel = {'model': bestModel,
                   'aic': incumbentError,
                   'p': bestModel.k_ar,
                   'd': bestModel.k_diff,
                   'q': bestModel.k_ma}

    return figure_forecast, figure_residuals, resultModel


def ARIMAfit(series: pd.Series,
             p: int,
             d: int,
             q: int) -> bool:
    """

    :param series: DESCRIPTION input pandas series to fit
    :type series: pd. series
    :param p: DESCRIPTION ARIMA parameter P
    :type p: int
    :param d: DESCRIPTION ARIMA parameter D
    :type d: int
    :param q: DESCRIPTION ARIMA parameter Q
    :type q: int
    :return: DESCRIPTION
    :rtype: bool

    """

    model = ARIMA(series, order=(p, d, q))
    results_AR = model.fit(disp=-1)
    plt.plot(series)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.title('ARIMA fit p=' + str(p) + ' q=' + str(q) + ' d=' + str(d))

    # Plot output figure
    plt.figure()
    results_AR.plot_diagnostics(figsize=(15, 12))
    return True


def autoSARIMAXfit(y, minRangepdq, maxRangepdqy, seasonality):
    minRangepdq = np.int(minRangepdq)
    maxRangepdqy = np.int(maxRangepdqy)
    seasonality = np.int(seasonality)

    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(minRangepdq, maxRangepdqy)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], seasonality) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore")  # specify to ignore warning messages

    incumbentError = 9999999999
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                if(results.aic < incumbentError):
                    bestModel = mod
                    incumbentError = results.aic

                # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except Exception:
                continue
    return bestModel


def forecastSARIMAX(series: pd.Series,
                    minRangepdq: int,
                    maxRangepdqy: int,
                    seasonality: int,
                    NofSteps: int,
                    title: str) -> tuple:
    """
    the function test several consecutive values of (p, d, q) using SARIMAX model fitting.
    :param series: DESCRIPTION input pandas series to fit
    :type series: pd.series
    :param minRangepdq: DESCRIPTION minimum value among  (p, d, q) to test
    :type minRangepdq: int
    :param maxRangepdqy: DESCRIPTION maximum value among  (p, d, q) to test
    :type maxRangepdqy: int
    :param seasonality: DESCRIPTION value of seasonality
    :type seasonality: int
    :param NofSteps: DESCRIPTION number of future time points to forecast
    :type NofSteps: int
    :param title: DESCRIPTION title of the output figure
    :type title: str
    :return: DESCRIPTION output tuple
    :rtype: tuple

    """

    NofSteps = np.int(NofSteps)
    # residui=plt.figure()
    result = autoSARIMAXfit(series, minRangepdq, maxRangepdqy, seasonality)
    results = result.fit()
    residui = results.plot_diagnostics(figsize=(15, 12))

    forecast = plt.figure()
    pred = results.get_prediction(start=len(series) - 1,
                                  end=len(series) + NofSteps,
                                  dynamic=True)
    pred_ci = pred.conf_int()

    ax = series.plot(label='observed', color='orange')
    pred.predicted_mean.plot(ax=ax, label='Dynamic forecast', color='r', style='--', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='y', alpha=.2)

    ax.set_xlabel('Timeline')
    ax.set_ylabel('Series value')
    plt.title('Forecast: ' + title)
    plt.legend()
    return residui, forecast


def fourierAnalysis(y: np.array) -> pd.DataFrame:
    """
    The function applies the fast Fourier transform to a time series and returna a pandas DataFrame with the significant
    fourier Coefficients
    :param y: DESCRIPTION input array of float
    :type y: np.array
    :return: DESCRIPTION
    :rtype: TYPE pd.DataFrame

    """

    y = y.reshape(len(y),)
    N = len(y)
    T = 1  # assume having one sample for each time period

    t = np.arange(0, len(y)).reshape(len(y),)
    p = np.polyfit(t, y, 1)         # find linear trend in x
    y_notrend = y - p[0] * t

    # calculate fourier transform
    yf = np.fft.fft(y_notrend)

    # filter on the most significant coefficients (frequencies explaining at least 10% of the seasonality)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    amplitude = 2.0 / N * np.abs(yf[0:N // 2])
    weeks = 1 / xf

    data = {'Frequency_domain_value': xf,
            'Time_domain_value': weeks,
            'Amplitude': amplitude}
    D = pd.DataFrame(data)
    D = D.replace([np.inf, -np.inf], np.nan)
    D = D.dropna()
    D = D.sort_values(['Amplitude'], ascending=False)
    D['perc'] = D['Amplitude'] / np.sum(D['Amplitude'])
    D['cumsum'] = D['perc'].cumsum()

    return D


def transformSeriesToStationary(series: pd.Series,
                                signifAlpha: float = 0.05) -> tuple:
    """
    this function tries log, power and square root transformation to stationary series
    it returns the series and a string with the model used to transform the series
    reference: http://www.insightsbot.com/blog/1MH61d/augmented-dickey-fuller-test-in-python

    :param series: DESCRIPTION pandas series to transform stationary
    :type series: pd.Series
    :param signifAlpha: DESCRIPTION, defaults to 0.05. significance level (0.1 , 0.05, 0.01) to accept or reject the null hypothesis of Dickey fuller
    :type signifAlpha: float, optional
    :return: DESCRIPTION
    :rtype: tuple

    """

    def _returnPandPstar(result):
        p_value = result[1]

        p_star = signifAlpha

        # in alternativa si puo' usare il valore della statistica del test e i valori critici
        '''
        if signifAlpha==0.01:
            p_star=result[4]['1%']
        elif signifAlpha==0.05:
            p_star=result[4]['5%']
        if signifAlpha==0.1:
            p_star=result[4]['10%']
        '''

        return p_value, p_star

    ###########################################################################
    # test the original series
    result = adfuller(series, autolag='AIC')
    p_value, p_star = _returnPandPstar(result)

    '''
    If the P-Value is less than the Significance Level defined,
    we reject the Null Hypothesis that the time series contains a unit root.
    In other words, by rejecting the Null hypothesis,
    we can conclude that the time series is stationary.
    '''

    if (p_value < p_star):
        print("The initial series is stationary")
        model = 'initial'
        return series, model

    ###########################################################################
    # trying with power transformation
    series_transformed = series**2

    result = adfuller(series_transformed, autolag='AIC')
    p_value, p_star = _returnPandPstar(result)

    if (p_value < p_star):
        print("The transformed series using POWER transformation is stationary")
        model = 'POWER:2'
        return series_transformed, model

    ###########################################################################
    # trying with square root transformation
    series_transformed = np.sqrt(series)

    result = adfuller(series_transformed, autolag='AIC')
    p_value, p_star = _returnPandPstar(result)

    if (p_value < p_star):
        print("The transformed series using SQUARE ROOT transformation is stationary")
        model = 'SQRT'
        return series_transformed, model

    ###########################################################################
    # trying with logarithm transformation
    series_temp = series + 0.001
    series_transformed = np.log(series_temp)

    result = adfuller(series_transformed, autolag='AIC')
    p_value, p_star = _returnPandPstar(result)

    if (p_value < p_star):
        print("The transformed series using LOG transformation is stationary")
        model = 'LOG'
        return series_transformed, model

    ###########################################################################
    # trying with boxcox transformation
    series_transformed, lam = boxcox(series_temp)

    result = adfuller(series_transformed, autolag='AIC')
    p_value, p_star = _returnPandPstar(result)

    if (p_value < p_star):
        print("The transformed series using BOXCOX, lambda:{lam} transformation is stationary")
        model = f"BOXCOX, lambda:{lam}"
        return series_transformed, model

    print("No valid transformation found")
    return [], []


def attractor_estimate(y, dim='3d') -> bool:
    """
    Uses the Ruelle & Packard method to estimate an attractor
    :param y: DESCRIPTION time series to evaluate
    :type y: TYPE
    :param dim: DESCRIPTION, defaults to '3d'. '3d' or '2d' projection
    :type dim: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    if dim not in ['2d', '3d']:
        raise ValueError(f"""dim parameter: {dim} not a valid projection.
                          Choose between ['2d', '3d']""")
    # TODO: add the time lag choice
    output_fig = {}

    #  Ruelle & Packard reconstruction
    y_2 = y[1:]
    y_3 = y[2:]

    # fix array length
    y = y[:len(y_3)]
    y_2 = y_2[:len(y_3)]

    if dim == '3d':
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(y, y_2, y_3, lw=0.5)
        plt.title(f" {dim} attractor estimate")
        output_fig['attractor_fig'] = fig
    elif dim == '2d':
        fig = plt.figure()
        plt.plot(y, y_2, lw=0.5)
        plt.title(f" {dim} attractor estimate")
        output_fig['attractor_fig'] = fig
    else:
        print("Choose 3d or 2d dimension")
    return True


def poincare_section(series: pd.Series,
                     T: int = 2,
                     num_of_dots_on_picture: int = 10) -> tuple:
    """
    Define the poincare section of a time series at time lags T and output
    a figure for each time lag containing a given number of dots
    :param series: DESCRIPTION time series to analyse
    :type series: TYPE
    :param T: DESCRIPTION, defaults to 2. time lag at which evaluate the time series
    :type T: TYPE, optional
    :param num_of_dots_on_picture: DESCRIPTION, defaults to 10. number of dots for each image of the poincare section
    :type num_of_dots_on_picture: TYPE, optional
    :return: DESCRIPTION pandas dataframe with poincare section coordinates for each time lag evaluated,
                         corresponding predicted value (next time lag()) and an image (rgb array) with the
                         num_of_dots_on_picture poincare section evaluated at that step

                         dictionary containing the poincare section at the last time lag
    :rtype: TYPE tuple

    """

    # create an output dictionary for figures
    out_fig = {}

    # create a dataframe with coordinates of the poincare section
    # the corrensponding predicting value
    D_all_coords = pd.DataFrame(columns=['x_coord', 'y_coord', 'value_to_predict'])

    # define the poincare section at each time lag
    for i in range(T, len(series) - 1):
        poincare_new_coord = (series[i], series[i - T], series[i + 1])
        D_all_coords = D_all_coords.append(pd.DataFrame([poincare_new_coord],
                                                        columns=['x_coord', 'y_coord', 'value_to_predict']))

    # set progressive index
    D_all_coords.index = list(range(0, len(D_all_coords)))

    # plot Poincare Section of the Time series with the given Time Lag

    # set colors
    c_list = list(range(len(D_all_coords)))
    cmap = cm.autumn
    norm = Normalize(vmin=min(c_list), vmax=max(c_list))

    # define the figure
    fig = plt.figure()
    plt.scatter(D_all_coords['x_coord'], D_all_coords['y_coord'], s=0.5, c=cmap(norm(c_list)))
    plt.title(f"Poincare section with k={T}")
    out_fig['PoincareSection'] = fig

    # output the image arrays for predictions

    # add a column for the images with the poincare sections
    D_all_coords['PoincareMaps'] = ''
    for position in range(0, len(D_all_coords)):

        beginning = max(0, position - num_of_dots_on_picture)
        end = position + 1
        plt.scatter(D_all_coords['x_coord'].iloc[beginning:end], D_all_coords['y_coord'].iloc[beginning:end], s=0.5, c='black')
        plt.xlim((min(D_all_coords['x_coord']), max(D_all_coords['x_coord'])))
        plt.ylim((min(D_all_coords['y_coord']), max(D_all_coords['y_coord'])))
        plt.axis('off')
        out_fig['PoincareSection'] = fig
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        D_all_coords.at[position, 'PoincareMaps'] = data

    return D_all_coords, out_fig
