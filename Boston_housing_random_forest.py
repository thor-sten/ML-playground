"""
Random Forest feature importance implementation on Boston housing dataset

506 rows and 14 columns.
crim:       Per capita crime rate by town.
zn:         Proportion of residential land zoned for lots over 25,000 sq.ft.
indus:      Proportion of non-retail business acres per town.
chas:       Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
nox:        Nitrogen oxides concentration (parts per 10 million).
rm:         Average number of rooms per dwelling.
age:        Proportion of owner-occupied units built prior to 1940.
dis:        Weighted mean of distances to five Boston employment centres.
rad:        Index of accessibility to radial highways.
tax:        Full-value property-tax rate per $10,000.
ptratio:    Pupil-teacher ratio by town.
b:          1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
lstat:      Lower status of the population (percent).
medv:       Median value of owner-occupied homes in $1000s.
"""

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load data
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

rf = RandomForestRegressor()
rf.fit(X, Y)

print('Features sorted by mean decrease impurity score:')
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))
# sorted(zip(np.round_(rf.feature_importances_, 4), names), reverse=True)
