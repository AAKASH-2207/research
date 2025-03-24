import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("2024_complete_dataset.csv")
data.head()
data = data.dropna()
print(len(data))
di = {"district":data['District'],"Data":data['Date'],"15cm":data['Avg_smlvl_at15cm']}
dataset = pd.DataFrame(di)

x = data['District']
y = data['Avg_smlvl_at15cm']

print(dataset.describe())
