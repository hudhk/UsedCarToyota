import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('used_toyota_data.csv')
plt.figure(figsize=(12, 8))

# Assuming 'Price' is our dependent variable on 'Mileage", 'Age', 'Horsepower', and 'Weight' as independent
num_vars = ['Mileage', 'Age', 'Horse Power', 'Weight']
for i, var in enumerate(num_vars, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[var], bins=20, kde=True)
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')

# Distribution of dependent variable 'Price'

plt.subplot(2, 2, 4)
sns.histplot(df['Price'], bins=20, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Pairplot to see relationships between Variables

sns.pairplot(df[num_vars + ['Price']])
plt.show()

# Correlation Matrix

corr_matrix = df[num_vars + ['Price']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Absolute Correlation Coefficients with 'Price'

corr_w_price = corr_matrix['Price'].abs().sort_values(ascending=False)
print(corr_w_price)

# Select Top 3 or 4 Most Important Variables

top_feat = corr_w_price[1:5].index.tolist()
print(top_feat)

#Converting 'Fuel Type' to Dummy Var

df = pd.get_dummies(df, columns=['Fuel Type'], prefix='Fuel Type', drop_first=True)

x = df.drop('Price', axis = 1)
y = df['Price']

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

reg_model = LinearRegression()
reg_model.fit(x_train, y_train)

y_train_pred = reg_model.predict(x_train)
y_val_pred = reg_model.predict(x_val)
y_test_pred = reg_model.predict(x_test)

residuals_train = y_train - y_train_pred
residuals_val = y_val - y_val_pred
residuals_test = y_test - y_test_pred

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(residuals_train, bins=20, kde=True)
plt.title('Training Set Error Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
sns.histplot(residuals_val, bins=20, kde=True)
plt.title('Validation Set Error Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
sns.histplot(residuals_test, bins=20, kde=True)
plt.title('Test Set Error Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()