from bin import get_prices as hist
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tests.preprocessing import DataProcessing

start = "2017-01-01"
end = "2020-06-01"
look_back = 30
n_features = 1

lstm_units = 32
dropout_prob = 0.1
epochs = 100
batch_size = 8
optimizer = "adam"


hist.get_stock_data("PETR4.SA", start_date=start, end_date=end)

process = DataProcessing("stock_prices.csv", 0.9)
process.gen_test(look_back)
process.gen_train(look_back)

X_train = process.X_train/10
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
Y_train = process.Y_train/10

X_test = process.X_test/10
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
Y_test = process.Y_test/10

# print(X_test)
# print(Y_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=lstm_units, return_sequences=True, input_shape=(look_back, n_features)))
model.add(tf.keras.layers.Dropout(dropout_prob))
model.add(tf.keras.layers.LSTM(units=lstm_units))
model.add(tf.keras.layers.Dropout(dropout_prob))

model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

# Compile and fit the LSTM network
print("Compile...")
model.compile(loss='mean_squared_error', optimizer=optimizer)

print("Fit...")
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)

print("Evaluate:")
trainScore = model.evaluate(X_train, Y_train)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(X_test, Y_test)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# generate predictions for training

print("\nPredict...")
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

plt.plot(Y_train*10)
plt.plot(trainPredict*10)
plt.plot(Y_test*10)
plt.plot(testPredict*10)
plt.show()
