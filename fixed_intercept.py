from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import matplotlib.pyplot as plt

class FixedInterceptRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, intercept=0.0):
        self.intercept = intercept
        self.slope_ = None

    def fit(self, X, y):
        # Calculate the slope with the fixed intercept
        X = X.ravel()

        self.slope_ = np.sum((X - X.mean()) * (y - self.intercept)) / np.sum((X - X.mean()) ** 2)
        return self

    def predict(self, X):
        # Predict y using the calculated slope and fixed intercept
        return self.slope_ * X.ravel() + self.intercept

    def get_params(self, deep=True):
        return {"intercept": self.intercept}

    def set_params(self, **params):
        self.intercept = params.get("intercept", self.intercept)
        return self



# Example data
X = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 2.2, 3.1, 3.9, 5.3, 6.1])

# Set the fixed intercept value
fixed_intercept = 0.5

# Create an instance of the custom regressor
fixed_intercept_regressor = FixedInterceptRegressor(intercept=fixed_intercept)

# Use RANSAC with the custom regressor
ransac = RANSACRegressor(estimator=fixed_intercept_regressor, min_samples=2, residual_threshold=0.5, max_trials=1000)
ransac.fit(X, y)

# Get the estimated slope
estimated_slope = ransac.estimator_.slope_
print(f"Estimated slope: {estimated_slope}")
print(f"Fixed intercept: {fixed_intercept}")

# Get the inliers mask
inlier_mask = ransac.inlier_mask_

# Predict using the fitted RANSAC model
line_X = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
line_y_ransac = ransac.predict(line_X)

# Plot the results
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
plt.scatter(X[~inlier_mask], y[~inlier_mask], color='gold', marker='.', label='Outliers')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', label='Fitted Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend(loc='best')
plt.show()
