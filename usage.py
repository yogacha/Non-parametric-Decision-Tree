# %%
from sklearn.datasets import load_iris
from tree import TauTree


data = load_iris()
X = data['data']
y = data['target']


regressor = TauTree(depth=2, min_split=2)
regressor.fit(X, y)
regressor.predict(X)

