import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("2025_complete_dataset.csv")

an_value = float("NaN")
data.replace(to_replace = 0.0, value= np.nan,inplace=True)

data = data.dropna()
print(data.head())