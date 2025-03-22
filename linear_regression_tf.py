import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow import Sequential
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras import regularizers

data = pd.read_csv("2024_complete_dataset.csv")
#d2 = pd.read_csv("Daily_data_of_Soil_Moisture_during_April_2024.csv")

#data = pd.concat([d1,d2],axis = 0)

data = data.dropna()
print(len(data))
x = data[['District','Date']]
y = data['Avg_smlvl_at15cm']

for col in x.columns:
    x = pd.get_dummies(x, columns=[col], drop_first=True)

x_train,x_test, y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=32)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


model = tf.keras.Sequential([tf.keras.layers.Dense(128, activation = 'relu',kernel_regularizer=regularizers.l1(0.00456)),
                             tf.keras.layers.Dropout(0.04505),
                       tf.keras.layers.Dense(128, activation = 'relu'),
                       tf.keras.layers.Dropout(0.04505),
                       tf.keras.layers.Dense(128, activation = 'relu'),
                       tf.keras.layers.Dropout(0.04505),
                       tf.keras.layers.Dense(128, activation = 'relu'),
                       tf.keras.layers.Dropout(0.04505),
                       tf.keras.layers.Dense(128, activation = 'relu'),
                       tf.keras.layers.Dropout(0.04505),
                       tf.keras.layers.Dense(1)
])
#0.000254
opti = tf.keras.optimizers.Adam(learning_rate = 0.000254)

model.compile(optimizer = opti,loss = "mean_squared_error")

es = 200

history = model.fit(x_train, y_train, validation_split = 0.2, epochs = es, batch_size = 1024)

model.save
test_loss = model.evaluate(x_test, y_test)

min_loss = min(history.history['loss'])
print(f"minimum loss = {min_loss}")
test_accuracy = 100-test_loss
print(f"test accuracy = {test_accuracy}%")

print(f'test loss :{test_loss}')

predictions = model.predict(x_test)

#model.save("satatalitte_data_soil_moisture.h5")
#print("model saved")

#plotting actual v/s predicted values
plt.scatter(y_test, predictions, marker='.')

plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.title('actual v/s predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

#plotting the loss over interartions
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()