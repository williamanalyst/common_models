# np.exp(x) = e^x
# Dealing with categorical data (reults)
""" Logistic Regression is about output = Probablity """
# Probablity(x) = e^(a0 + a1*x1 + a2*x2 + a3*x3 + ...) / (1 + e^ (a0 + a1*x1 + a2*x2 + a3*x3 + ...) )
# or Odds =  Probablity(x)/ (1 - Probablity(x)) = e ^ (a0 + a1*x1 + a2*x2 + a3*x3 + ...)
# or Log (Odds) = a0 + a1*x1 + a2*x2 + a3*x3 + ... (Preferred Form)
#
""" Assumption for Logistic Regression is very similar to Linear Regression:
    1. Non-Linear
    2. No En
""" 
# In[]: 
#
import numpy as np
import pandas as pd
pd.options.display.max_rows = 20
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import os
os.chdir('C:\Python_Project\Logistic_Regression\Data')
#
df = pd.read_csv('Simple_Logistic.csv')
df['Admitted'] = df['Admitted'].map({'Yes': 1, 'No': 0})
#
df
y = df['Admitted']
x1 = df['SAT']
#
plt.scatter(x1, y, color = 'C0')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('Admitted', fontsize = 20)
#
x = sm.add_constant(x1)
reg_lin = sm.OLS(y, x)
results_lin = reg_lin.fit()
plt.scatter(x1, y, color = 'C0')
y_hat = x1*results_lin.params[1] + results_lin.params[0]
plt.plot(x1, y_hat, lw = 2.5, color = 'red')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('Admitted', fontsize = 20) 
# Non-linear relationship makes it un-fit for linear regression
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()
#
results_log.summary() # 
#
def f(x, b0, b1):
    return np.array(np.exp(b0 + x * b1)/ (1 + np.exp(b0 + x * b1 )))
f_sorted = np.sort(f(x1, results_log.params[0], results_log.params[1]))
x_sorted = np.sort(np.array(x1))
#
plt.scatter(x1, y, color = 'r')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('Admitted', fontsize = 20)
plt.plot(x_sorted, f_sorted, color = 'b')
#
# In[]:
#
y = df['Admitted']
x1 = df['SAT']
#
x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit() # maximum number of iteration is 35
results_log.summary() # 
#
# stats model is using a function from scipy
    # When that function is removed from scipy, this will cause stats model from working
    # And the function is actually:
    # from scipy import stats
    # stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
#
""" Maximum Likelihood Estimation """ # The bigger the likelihood fuction, the higher probablity our model is correct.
""" Log-Likelihood """ # Almost always negative, the smaller, the better
""" LL-Null = Log-Likelihood Null """ # The log-likelihood of a model which has no independent variables, y = a0 * 1 (constant)
# example:
x0 = np.ones(168)
reg_log = sm.Logit(y, x0)
results_log = reg_log.fit()
results_log.summary()
""" LLR p-value """ # Measures if our model is statistically different from LL-null, a.k.a. a useless model
""" Pseudu R-squared """ # McFadden's R-squared: "A good Pseudu R-Squared is somewhere between 0.2 and 0.4"
# This measure is mostly useful for comparing variations of the same model.
# Different models will have completely different and incomparable Pseudu R-squared.
""" Coefficient """
# log( Probablity2 - Probablity1) = coefficient * (x2 - x1)
# Probablity2 - Probablity1 = e ^ (coefficient * (x2 - x1) )
# In[]: 
#
df = pd.read_csv('Logistic_with_Binary.csv')
df['Admitted'] = df['Admitted'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
#
y = df['Admitted']
#x1 = df['Gender']
x1 = df.drop(['Admitted'], axis = 1)
#
x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()
results_log.summary()
# Logit Results object has not attribute 'coef_'
np.set_printoptions(formatter = {'float': lambda x: '{0:0.2f}'.format(x)}, precision = 2)
#np.set_printoptions(precision = 2)
np.set_printoptions(suppress = True)
results_log.predict()
#
np.set_printoptions() # reset formatting to default
#
np.array(df['Admitted'])
#
results_log.pred_table() # Confusion Matrix 
#
cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index = {0: 'Actual 0', 1: 'Actual 1'})
cm_df
#
cm = np.array(cm_df)
accuracy_train = (cm[0,0] +cm[1,1])/ cm.sum()
accuracy_train
#
# In[]:
test = pd.read_csv('Binary_Test.csv')
test['Admitted'] = test['Admitted'].map({'Yes': 1, "No": 0})
test['Gender'] = test['Gender'].map({'Male': 1, 'Female': 0})
test
#
test_actual = test['Admitted']
test_data = test.drop(['Admitted'], axis = 1)
test_data = sm.add_constant(test_data)
test_data
#
def confusion_matrix(data, actual_values, model):
    pred_values = model.predict(data)
    bins = np.array([0, 0.5, 1])
    cm = np.histogram2d(actual_values, pred_values, bins = bins)[0]
    accuracy = (cm[0,0] + cm[1,1])/ cm.sum()
    return cm, accuracy
#
cm, accuracy = confusion_matrix(test_data, test_actual, results_log )
cm
accuracy
#
#
# In[]:
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('C:\Python_Project\Logistic_Regression\Data')
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# In[]:
# 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('C:\Python_Project\Logistic_Regression\Data')
# Importing the dataset
items = pd.read_csv('Exchange_Items.csv')
samples = pd.read_csv('Exchange_Samples.csv')
samples['created'] = pd.to_datetime(samples['created'])
#
samples_group = samples.groupby('item_id').agg({'id': np.count_nonzero, 'created':min})
item_sample = pd.merge(left = items, right = samples_group, left_on = 'id', right_on = 'item_id', how = 'left')
#
item_sample_selected = item_sample[['id_x', 'ref_price', 'vol_min', 'id_y']]
item_sample_selected.columns = ['listing_id', 'ref_price', 'vol_min', 'has_sample']
item_sample_selected['has_sample'] = item_sample_selected['has_sample'].fillna(value = 0).apply(lambda x: 1 if x >= 1 else 0)
item_sample_requested = item_sample_selected.loc[item_sample_selected['has_sample'] == 1]
item_sample_no = item_sample_selected.loc[item_sample_selected['has_sample'] == 0].sample(n = len(item_sample_requested))
item_sample_selected = pd.concat([item_sample_requested,item_sample_no], axis = 0 )
item_sample_selected = item_sample_selected.fillna(value = 0)
#
X = item_sample_selected.iloc[:, 1:3].values
y = item_sample_selected.iloc[:, 3].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Price')
plt.ylabel('Volume')
plt.legend()
plt.xlim(-1,5)
plt.ylim(-1,10)
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Price')
plt.ylabel('Volume')
plt.legend()
plt.xlim(-1, 1.5)
plt.ylim(-1, 3)
