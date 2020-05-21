""" Created on Mon Dec  9 14:34:38 2019 @author: willi """ 
# Principal Component Analysis
""" It is pertty difficult to visualise high-dimensional data, therefore,
    we can use PCA to find key principal components """
#
""" PCA 
    # 1. First, shift the centre all Feature_1 and Feature_2 data to (0, 0), as shifting whole dataset does not affect internal relationships.
    # 2. Then, find the linear combination between Feature_1 and Feature_2, and forms PC-1. (after getting the best-fitting line) 
        # PC1 = weight1 * Feature_1 + weight2 * Feature_2 
        # weight1 **2 + weight2 **2 = 1 ( called as unit vector/ singular vector/ eigenvector for PC-1 )
    # 3. 
"""
#
# In[]:
# PCA is an unsupervised statistical technique used to example the interrelations among 
#     a set of variables in order to identify the underlying structure of those variables.
#     aka "general factor analysis"
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
#
from sklearn.datasets import load_breast_cancer
#
cancer = load_breast_cancer()
type(cancer)
cancer.keys()
#print(cancer['DESCR'])
#
df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
df.head()
cancer['target']
cancer['target_names']
# 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
# Actual PCA Process
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape
type(x_pca) # x_pca is a numpy array
#
plt.figure(figsize = (8,6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = cancer['target'], cmap = 'plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
#
pca.components_
#
df_comp = pd.DataFrame(pca.components_, columns = cancer['feature_names'])
df_comp
#
plt.figure(figsize = (12,6))
sns.heatmap(df_comp, cmap = 'plasma')
#
# Then use SVM to further analyze
#
# In[]:
#
#
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle
#
iris = load_iris()
numSamples, numFeatures = iris.data.shape
print(numSamples)
print(numFeatures)
print(list(iris.target_names))
# 
X = iris.data
pca = PCA(n_components = 2, whiten = True).fit(X) # whiten means to normalize all the data
X_pca = pca.transform(X) # Above process reduces the Dimension from 4 into 2 Dimension
#
print(pca.components_)
#
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_)) # 97.77% of the variance can actually be explained by the 2 variables compared to 4
#
colors = cycle('rgb')
target_ids = range(len(iris.target_names))
pl.figure()
for i, c, label in zip(target_ids, colors, iris.target_names):
    pl.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1], c=c, label = label)
pl.legend()
pl.show()
