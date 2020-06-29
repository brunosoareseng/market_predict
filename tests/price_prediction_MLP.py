from bin import get_prices as hist
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tests.preprocessing import DataProcessing

start = "2016-01-01"
end = "2020-01-01"
look_back = 10

hist.get_stock_data("PETR4.SA", start_date=start, end_date=end)

process = DataProcessing("stock_prices.csv", 0.9)
process.gen_test(look_back)
process.gen_train(look_back)

X_train = process.X_train
Y_train = process.Y_train

X_test = process.X_test
Y_test = process.Y_test

# print(X_test)
# print(Y_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, input_dim=look_back, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train, Y_train, epochs=100, verbose=0)

print("\n\nEvaluate:")
trainScore = model.evaluate(X_train, Y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(X_test, Y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# generate predictions for training
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

# shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(process.stock_train)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
# testPredictPlot = numpy.empty_like(process.stock_test)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(process.stock_test) - 1, :] = testPredict
# plot baseline and predictions

plt.plot(Y_train)
plt.plot(trainPredict)
plt.plot(Y_test)
plt.plot(testPredict)
plt.show()
