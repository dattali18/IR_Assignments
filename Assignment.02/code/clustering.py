#-------------------------------------------------------------------------
# Imports
# 1. numpy
# 2. matplotlib
# 3. KMeans, GMM, DBSCAN
#-------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import pandas as pd

import os

#-------------------------------------------------------------------------
# Constants
#-------------------------------------------------------------------------

CLUSTERS_COUNT = 4

DATA_DIR = os.path.join(os.path.dirname(__file__), '..',  'data')
DOC2VEC_DIR = os.path.join(DATA_DIR, 'doc2vec')

#-------------------------------------------------------------------------
# Doc2Vec Clustering
#-------------------------------------------------------------------------

def get_doc2vec_data():
    """
    The data frame is as follows:
    1. Sheet {A-J, BBC, J-P, NY-T} str
    2. RowIndex int
    3. Dim0 - Dim299 float
    """
    return pd.read_csv(os.path.join(DOC2VEC_DIR, 'doc2vec_vectors.csv'))