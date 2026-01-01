import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Simulate stock prices
np.random.seed(42)
days = np.arange(1, 51)
prices = 100 + 1.2 * days + 5 * np.sin(days / 5) + np.random.normal(0, 2.5, size=len(days))

# Linear Regression
X = days.reshape(-1, 1)
y = prices
linear_reg = LinearRegression()
linear_reg.fit(X, y)
y_linear_pred = linear_reg.predict(X)
r2_linear = r2_score(y, y_linear_pred)

# Polynomial Regression (degree=3)
degree = 3
poly_features = PolynomialFeatures(degree)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)
r2_poly = r2_score(y, y_poly_pred)

# Predict future prices for the next 10 days
future_days = np.arange(51, 61).reshape(-1, 1)
full_days = np.vstack((X, future_days))
# Linear prediction
future_linear_pred = linear_reg.predict(full_days)
# Polynomial prediction
full_days_poly = poly_features.transform(full_days)
future_poly_pred = poly_reg.predict(full_days_poly)

# Visualization and save to file
plt.figure(figsize=(12, 6))
plt.scatter(days, prices, color='black', label='Simulated True Price')
plt.plot(full_days.flatten(), future_linear_pred, '--r', label='Linear Regression')
plt.plot(full_days.flatten(), future_poly_pred, '-b', label=f'Polynomial Regression (degree={degree})')
plt.axvline(x=50, color='grey', linestyle=':', label='Start of Prediction')
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Simulated Stock Price & Regression Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("regression_stock.png")
plt.show()

# Print important results
print("=== Linear Regression Parameters ===")
print(f"  Coefficient (slope): {linear_reg.coef_[0]:.4f}")
print(f"  Intercept: {linear_reg.intercept_:.4f}")
print(f"  R-squared Score: {r2_linear:.4f}")
print()

print(f"=== Polynomial Regression (degree {degree}) Parameters ===")
print(f"  Coefficients: {poly_reg.coef_}")
print(f"  Intercept: {poly_reg.intercept_:.4f}")
print(f"  R-squared Score: {r2_poly:.4f}")
print()

print(f"{'Day':>3} | {'Linear_Pred':>12} | {'Poly_Pred':>12}")
for i, day in enumerate(range(51, 61)):
    print(f"{day:>3} | {future_linear_pred[len(X)+i]:12.2f} | {future_poly_pred[len(X)+i]:12.2f}")

print("\nPlot saved as 'regression_stock.png'")

"""
pip install matplotlib scikit-learn numpy
"""
