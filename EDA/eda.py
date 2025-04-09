import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("EDA/data.csv")

df_cleaned = df.drop(['Engine Fuel Type', 'Number of Doors', 'Market Category','Vehicle Size', 'Vehicle Style', 'Popularity'], axis=1)
df_cleaned = df_cleaned.rename(columns={'Engine HP':'HP', 'Engine Cylinders':'Cylinders', 'Transmission Type':'Transmission', 'Driven_Wheels':'Drive', 'highway MPG':'MPG H','city mpg':'MGP C', 'MSRP': 'Price'})

duplicate_r = df_cleaned[df_cleaned.duplicated]

df_cleaned = df_cleaned.drop_duplicates()

df_cleaned = df_cleaned.dropna()

num_col = df_cleaned.select_dtypes(include=["number"]).columns
cat_col = df_cleaned.select_dtypes(exclude=["number"]).columns

Q1 = df_cleaned[num_col].quantile(0.25)
Q3 = df_cleaned[num_col].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df_cleaned[~((df_cleaned[num_col] < lower_bound) | (df_cleaned[num_col] > upper_bound)).any(axis=1)]

df_cleaned.to_csv('EDA/new_data.csv')


# df_cleaned.Make.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
# plt.title("Number of cars by make")
# plt.ylabel('Number of cars')
# plt.xlabel('Make')
# plt.show()

correlation_matrix = df_cleaned[num_col].corr()
plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, cmap="BrBG", annot=True, fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

print(correlation_matrix)

