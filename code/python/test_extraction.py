from feature_extraction.tsfeature.feature_core import sequence_feature
import pandas as pd
import scipy.io
import numpy as np
from pca import pca
import matplotlib.pyplot as plt

from dtw import distance, dtw
p1 = [1,2,3,3,7,9,10,5,4,8,10]
p2 = [1,2,3,5,3,7,9,3,6,2,10,6]

print(dtw(p1, p2))

