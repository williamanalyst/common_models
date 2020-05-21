# In[]: Case_1
# Import the libraries
# Pandas is high-performance, easy-to-use data structures and data analysis tools for the Python programming language
import pandas as pd
# Numpy is the fundamental package for scientific computing with Python
import numpy as np
# Matplotlib is used for graphs
import matplotlib.pyplot as plt
# seaborn Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics
import seaborn as sns
# %matplotlib inline is magic command. This performs the necessary behind-the-scenes setup for IPython to work correctly hand in hand with matplotlib
%matplotlib inline
# Importing data set
from sklearn.datasets import make_blobs
#
data= make_blobs(n_samples=200, n_features= 2, centers=4, cluster_std=1.8, random_state=101)
#
data[0]
# Shape of data
data[0].shape
# Scatter plot of different blobs
plt.scatter(data[0][:,0],data[0][:,1],c=data[1])
data[1]
# Import Kmean Model
from sklearn.cluster import KMeans
# Specifying a variable to  KMeans()
kmeans = KMeans(n_clusters=4)
# Creating Clusters
kmeans.fit(data[0])
kmeans.cluster_centers_
# Checking labels
kmeans.labels_
# Visualizing our predicted clusters and original clusters
fig, (ax1,ax2)= plt.subplots(1,2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_)
ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1])
#
# In[]: Case_2
# Import the libraries
# Pandas is high-performance, easy-to-use data structures and data analysis tools for the Python programming language
import pandas as pd
# Numpy is the fundamental package for scientific computing with Python
import numpy as np
# Matplotlib is used for graphs
import matplotlib.pyplot as plt
# seaborn Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics
import seaborn as sns
# %matplotlib inline is magic command. This performs the necessary behind-the-scenes setup for IPython to work correctly hand in hand with matplotlib
%matplotlib inline
# Importing College_Data
import os
os.chdir('C:\Python_Project\Data_Collection')
df = pd.read_csv('College_Data.csv',index_col=0)
# To view first 5 rows of data set
df.head()
# To know data type and non null values in a data frame
df.info()
# Describe function provide descriptive statistics Output table of data 
df.describe()
#
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
# Import Kmean Model
from sklearn.cluster import KMeans
# Specifying a variable to  KMeans()
kmeans = KMeans(n_clusters=2)
# Creating Clusters
kmeans.fit(df.drop('Private',axis=1))
#
kmeans.cluster_centers_
# Function to convert cluster categorical variable to binary
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
#
df['Clusters']=0
#
df['Cluster'] = df['Private'].apply(converter)
#
df.head()
# Importing confusion_matrix & classification_report to verify our relust
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))
