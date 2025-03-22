import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


data = pd.read_csv("Daily_data_of_Soil_Moisture_during_March_2024.csv")
data = data.dropna()
print(len(data))
X = data[['District','Date']]
Y = data['Avg_smlvl_at15cm']

categorical_features = ['District','Date']
preprocessor = ColumnTransformer(transformers=[('cat',OneHotEncoder(),categorical_features)])


model = Pipeline(steps = [('preprocessor',preprocessor),('model', LinearRegression())])


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 32)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE=", mse)
print("r2 = ", r2)

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual V/s Predicted")
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
plt.show()

model1 = model.named_steps['model']
preprocessor = model.named_steps['preprocessor']

# Get feature names after One-Hot Encoding
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([ohe_feature_names])

# Print the coefficients
coefficients = model1.coef_
for feature, coef in zip(all_feature_names, coefficients):
    print(f'Feature: {feature}, Coefficient: {coef}')