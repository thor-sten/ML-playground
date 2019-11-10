# Recursive Feature Elimination
# https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_friedman1
from sklearn.svm import SVR

# Load the iris dataset
dataset = datasets.load_iris()
# Create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# Create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(dataset.data, dataset.target)

print(rfe.support_) # Mask of selected features
print(rfe.ranking_) # Selected features are assigned rank 1


# Feature importance: Fit an Extra Trees model to the data
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
model2 = ExtraTreesClassifier()
model2.fit(dataset.data, dataset.target)

# Display the relative importance of each attribute
print(model2.feature_importances_)


# Retrieve the 5 right informative features in the Friedman #1 dataset
X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X, y)

print(selector.support_)
print(selector.ranking_)