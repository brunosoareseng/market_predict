from math import sqrt
from numpy import concatenate
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

num_days_validate = 10

# Lê arquivo com dados historicos
# Prepara dataset para uso

dataset = read_csv('../dados/stock_prices_full.csv', index_col=0)
# manually specify column names
dataset.columns = ['high', 'low', 'open', 'close', 'volume', 'adj close']
dataset.index.name = 'date'
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('teste.csv')

# Lê dataset do arquivo
# Plota as variaveis
# load dataset
dataset = read_csv('teste.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 4, 5]
k = 1
# plot each column
# plt.figure()
# for group in groups:
#     plt.subplot(len(groups), 1, k)
#     plt.plot(values[:, group])
#     plt.title(dataset.columns[group], y=0.5, loc='right')
#     k += 1
# plt.show()


# Calcula delta e remove valor futuro
def calculate_delta(data):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in df.columns:
        cols.append(df[i])
        names += [(df.columns[i])]

    cols.append(df[5] - df.shift(1)[5])
    names += [('delta')]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    agg.dropna(inplace=True)
    agg.drop(columns=5)
    #convert in ttable
    agg = agg.values
    # ensure all data is float
    agg = agg.astype('float32')

    return agg


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
dataset = read_csv('teste.csv', header=0, index_col=0)
values = dataset.values                                # print(values)
dados_orig = values[:, [5]]                            # print(dados_orig)

# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')

# Inclui coluna com variação do dia
values = calculate_delta(values)

# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[7, 8, 9, 10, 11, 12]], axis=1, inplace=True)
print("Tabela reframed, para treinamento e verificação")
print(reframed.head())


# ---------------------------------------------------------------------------
# Define model
# Fit model
# ---------------------------------------------------------------------------

# split into train and test sets
values = reframed.values
n_train_days = np.size(values, 0)-num_days_validate
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print("\nShape dos dados de treino e teste")
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# ---------------------------------------------------------------------------
# Make prediction
# Evaluate model
# ---------------------------------------------------------------------------

# design network
model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# inv_y = test_y[-30:-1]
# inv_yhat = inv_yhat[-30:-1]
dados_orig = dados_orig[-num_days_validate:-1]
# print("Dados originais:")
# print(inv_y)
# print(dados_orig)

inv_y = inv_y - inv_y.mean()
inv_yhat = 10*(inv_yhat - inv_yhat.mean())

plt.subplot(3, 1, 1)
plt.plot(inv_y)
plt.axhline(y=0, linewidth=0.5)
plt.title('dados', y=0.5, loc='right')
plt.subplot(3, 1, 2)
plt.plot(inv_yhat)
plt.axhline(y=0, linewidth=0.5)
plt.title('prediction', y=0.5, loc='right')
plt.subplot(3, 1, 3)
plt.plot(dados_orig)
plt.title('Dado real', y=0.5, loc='right')

# plt.plot(inv_y, label='dados')
# plt.plot(inv_yhat, label='prediction')
# plt.ylim(-0.1, 0.1)
# plt.axhline(y=0, linewidth=0.25)
# plt.legend()
plt.show()
