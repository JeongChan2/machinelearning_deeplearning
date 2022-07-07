from tensorflow.keras import datasets
from sklearn.linear_model import LinearRegression
(train_data,train_label), (test_data, test_label)=\
datasets.boston_housing.load_data()
type(train_data)
model_lr = LinearRegression()
model_lr.fit(train_data,train_label)
prediction_lr = model_lr.predict(test_data)
error = sum(abs(prediction_lr-test_label))/102
print(error)
