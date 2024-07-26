import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 

file = pd.read_csv("E:/progamming/Machine learning/internship/housing2.csv")
file['basement'] = file['basement'].map({'yes': 1, 'no': 0})

sns.lmplot(x ="area", y ="price", data = file, order = 1, ci = None) 
plt.show()
sns.lmplot(x ="bedrooms", y ="price", data = file, order = 1, ci = None) 
plt.show()
sns.lmplot(x ="bathrooms", y ="price", data = file, order = 1, ci = None) 
plt.show()
sns.lmplot(x ="stories", y ="price", data = file, order = 1, ci = None) 
plt.show()
sns.lmplot(x ="parking", y ="price", data = file, order = 1, ci = None) 
plt.show()
sns.lmplot(x ="basement", y ="price", data = file, order = 1, ci = None) 
plt.show()

Y = np.array(file['price']).reshape(-1, 1) 
X = np.array(file[['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'basement']])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3) 

model = LinearRegression()
model.fit(X_train, y_train)

print(f"The Coefficients are: {model.coef_}")
print(f"The y intercept: {model.intercept_}")
print("Score: ", model.score(X_test, y_test))

y_pred = model.predict(X_test)

feature_index = file.columns.get_loc('area')
plt.scatter(X_test[:, feature_index], y_test, color='b', label='Actual')
plt.scatter(X_test[:, feature_index], y_pred, color='k', label='Predicted')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()
