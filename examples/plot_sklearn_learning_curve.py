"""
================================
Learning Curve with scikit-learn
================================

Here we use sklearn's learning curve to explore the impact of the training
set size. The GaussianMixtureRegressor could also be used for model selection
with sklearn.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from gmr.sklearn import GaussianMixtureRegressor
from sklearn.datasets import fetch_california_housing


X, y = fetch_california_housing(return_X_y=True)
random_state = np.random.RandomState(0)
train_sizes_abs, train_scores, test_scores = learning_curve(
    GaussianMixtureRegressor(n_components=3, random_state=random_state),
    X, y, cv=3, verbose=2, n_jobs=-1, random_state=random_state)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Learning Curve with GaussianMixtureRegressor")
plt.xlabel(r"Training set size")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(train_sizes_abs, train_scores_mean, label="Training score",
         color="darkorange", lw=lw)
plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(train_sizes_abs, test_scores_mean, label="Cross-validation score",
         color="navy", lw=lw)
plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
