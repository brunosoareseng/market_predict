from math import sqrt
from numpy import concatenate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow import nn
from datetime import date
import datetime as dt


def prediction(papel):
    """
    Lê arquivos com preços e gerencia predição de preços gerando arquivos e graficos
    :param papel: Nome da ação para pegar arquivo com dados historicos do papel para simular
    :type papel: string nome da ação
    """

    num_days_validate = 10

    # Lê arquivo com dados historicos
    # Prepara dataset para uso

    dataset = read_csv("./dados/" + papel + '_dados.csv', index_col=0)

    # manually specify column names
    dataset.columns = ['high', 'low', 'open', 'close', 'volume', 'adj close']
    dataset.index.name = 'date'

    # save to file
    # dataset.to_csv('teste.csv')

    # Preparar dados para a LTSM
    # convert series to supervised learning
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # load dataset
    # dataset = read_csv('teste.csv', header=0, index_col=0)
    values = dataset.values  # print(values)
    index = dataset.index

    # integer encode direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # specify the number of lag days
    num_days_lag = 1
    n_features = 6
    # frame as supervised learning
    reframed = series_to_supervised(scaled, num_days_lag, 1)

    # print(reframed)
    # drop columns we don't want to predict
    # reframed.drop(reframed.columns[[6, 7, 8, 9, 10]], axis=1, inplace=True)

    # ---------------------------------------------------------------------------
    # Define model
    # Fit model
    # ---------------------------------------------------------------------------

    # split into train and test sets
    values = reframed.values

    n_train_days = np.size(values, 0) - num_days_validate
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    data_test = index[-num_days_validate:]
    data_test = data_test.values

    # split into input and outputs
    n_obs = num_days_lag * n_features
    train_x, train_y = train[:, :n_obs], train[:, -n_features]
    test_x, test_y = test[:, :n_obs], test[:, -n_features]
    pred_x = test[:, n_features:n_features + n_obs]

    # print("Train shapes x and Y")
    # print(train_x.shape, len(train_x), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], num_days_lag, n_features))
    test_x = test_x.reshape((test_x.shape[0], num_days_lag, n_features))
    pred_x = pred_x.reshape((pred_x.shape[0], num_days_lag, n_features))

    # ---------------------------------------------------------------------------
    # Make prediction
    # Evaluate model
    # ---------------------------------------------------------------------------

    # design network
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(LSTM(32, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1, activation=nn.relu))
    model.compile(loss='mean_squared_error', optimizer='nadam')
    # fit network
    history = model.fit(train_x, train_y, epochs=120, batch_size=32, validation_data=(test_x, test_y), verbose=2,
                        shuffle=False)

    # make a prediction
    yhat = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], num_days_lag * n_features))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_x[:, -5:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_x[:, -5:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # Previsão
    previsao = model.predict(pred_x)
    pred_x = pred_x.reshape((test_x.shape[0], num_days_lag * n_features))
    # invert scaling for forecast
    inv_previsao = concatenate((previsao, pred_x[:, -5:]), axis=1)
    inv_previsao = scaler.inverse_transform(inv_previsao)
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_previsao = inv_previsao[:, 0]
    inv_previsao = np.concatenate([[inv_yhat[0]], inv_previsao])  # adciona o primeiro elemento a previsao

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    # Prepare and plot
    # x axis with dates
    x = data_test

    # Plot history
    plt.subplot(2, 1, 1)
    titulo = papel + " - Test RMSE:" + "{:.2f}".format(rmse)
    plt.title(titulo)
    plt.grid(False)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label="test")
    plt.legend()

    # Plot resultado teste
    plt.subplot(2, 1, 2)
    plt.grid(True)
    plt.plot(x, inv_y, label='Valor real')
    plt.plot(inv_previsao, label='Previsão')
    plt.xticks(rotation=90)
    # plt.xlim(0, num_days_validate + 3 - 1)
    plt.legend()

    # Plot save
    # plt.show()
    plt.savefig("./resultado/Previsão_"+str(date.today())+"_"+papel+".png", dpi=600)
    plt.clf()
