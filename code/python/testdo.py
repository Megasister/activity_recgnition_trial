import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load dataset
X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names, index=load_iris().target)
print(X)

# Load pca
from pca import pca

# Initialize to reduce the data up to the nubmer of componentes that explains 95% of the variance.
# model = pca(n_components=0.95)

# Reduce the data towards 3 PCs
model = pca(n_components=3)

# Fit transform
results = model.fit_transform(X)
print(results)

fig, ax = model.scatter()
plt.show()