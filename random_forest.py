import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

data = pd.read_csv("2025_complete_dataset.csv")
data = pd.read_csv("2024_complete_dataset.csv")
#d2 = pd.read_csv("Daily_data_of_Soil_Moisture_during_April_2024.csv")

#data = pd.concat([d1,d2],axis = 0)
data.replace(to_replace = 0.0, value= np.nan,inplace=True)
data = data.dropna()

print(len(data))
x = data[['District','Date']]
y = data['Avg_smlvl_at15cm']

label_encoder = LabelEncoder()
x_categorical = data.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical = data.select_dtypes(exclude=['object']).values
x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values

x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 42)

regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

regressor.fit(x_train, y_train)

oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

predictions = regressor.predict(x_test)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, predictions)
print(f'R-squared: {r2}')

plt.scatter(y_test, predictions, marker='.')

plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.title('actual v/s predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()
from sklearn.tree import plot_tree


# Assuming regressor is your trained Random Forest model
# Pick one tree from the forest, e.g., the first tree (index 0)
tree_to_plot = regressor.estimators_[0]

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree_to_plot, feature_names=data.columns.tolist(), filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest")
plt.show()
