from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import matplotlib.pyplot as plt


class FixedSlopeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, slope=1.0):
        self.slope = slope
        self.intercept_ = None

    def fit(self, X, y):
        # Calculate the intercept based on the fixed slope
        self.intercept_ = np.mean(y - self.slope * X.ravel())
        return self

    def predict(self, X):
        # Use the fixed slope and calculated intercept to predict
        return self.slope * X.ravel() + self.intercept_

    def get_params(self, deep=True):
        return {"slope": self.slope}

    def set_params(self, **params):
        self.slope = params.get("slope", self.slope)


if __name__ == "__main__":
    # Example data
    X = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([1, 2, 2.8, 4.1, 5, 5.9])

    # Set the fixed slope value
    fixed_slope = 1.0

    # Create an instance of the custom regressor
    base_regressor = FixedSlopeRegressor(slope=fixed_slope)

    # Use RANSAC with the custom regressor
    ransac = RANSACRegressor(estimator=base_regressor, min_samples=2, residual_threshold=0.5, max_trials=1000)
    ransac.fit(X, y)

    # Get the inliers mask
    inlier_mask = ransac.inlier_mask_

    # Estimated intercept
    print(f"Estimated intercept: {ransac.estimator_.intercept_}")

    # Predict using the fitted RANSAC model
    line_X = np.arange(X.min(), X.max() + 1)[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    # Plot the results
    plt.scatter(X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers")
    plt.scatter(X[~inlier_mask], y[~inlier_mask], color="gold", marker=".", label="Outliers")
    plt.plot(line_X, line_y_ransac, color="cornflowerblue", label="RANSAC regressor")
    plt.legend(loc="best")
    #    plt.show()
    plt.savefig("fixed_gradient.png")
