import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 

df = pd.read_csv('real_estate_dataset.csv')
print(df.head())
print(df.duplicated())
print(df.isnull().sum())
print(df.describe())

fig, axs = plt.subplots(12,1,dpi=95, figsize=(7,17))
i = 0
for col in df.columns:
    axs[i].boxplot(df[col], vert=False)
    axs[i].set_ylabel(col)
    i+=1
plt.show()

#correlation
corr = df.corr()
 
plt.figure(dpi=130)
sns.heatmap(df.corr(), annot=True, fmt= '.2f')
plt.show()

print(corr['Price'].sort_values(ascending = False))

# Selecting features that are impacting prices
X = df.filter(['Num_Bedrooms', 'Square_Feet', 'Year_Built'])
df['Price_Log'] = np.log1p(df['Price'])
Y = df['Price_Log']

# learning the statistical parameters for each of the data and transforming
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
print(rescaledX[:5])

class LinearRegression:
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)

        # Calculate gradients
        dW = -2 * (self.X.T @ (self.Y - Y_pred)) / self.m
        db = -2 * np.sum(self.Y - Y_pred) / self.m

        # Update weights
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def predict(self, X):
        return X @ self.W + self.b

# Scale the features
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=1/3, random_state=0)

# Train the model
model = LinearRegression(iterations=1000, learning_rate=0.01)
model.fit(X_train, Y_train)

# Predict and evaluate
Y_pred = model.predict(X_test)
print("Predicted values:", Y_pred[:5])
print("Real values:", Y_test[:5])

from sklearn.metrics import mean_squared_error, r2_score

# Calculate Mean Squared Error and R^2 Score
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"MSE: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")