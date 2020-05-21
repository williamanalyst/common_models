# 5 tests must be done before doing an OLS Estimation
""" # 1. Linearity (can check visually) """
# Scatter plot can be used, sometimes log conversion can help finding linear relationship
#
###############################################################################
""" # 2. No Multicollinearity in independent variables (need to test correlation, can use correlation-matrix) """
# Use Correlationship Matrix, to test whether corr < 0.6 or < 0.8
    # Can also use scatter plot to show relationship between variables, e.g. use pairplot
    # Can use 'Tolerance', note that 'Tolerance =  (1 - R-squared)'
    # Can use Variance Inflation Factor (VIF) , note that ' VIF = 1/ Tolerance = 1 / (1 - R-squared) '
#
###############################################################################
""" # 3. Normality (of the residual/ error term) """ 
# Can use Q-Q plot to test whether the residuals are normally distributed
# Can plot a histogram of the residual
#
###############################################################################
""" # 4. Residuals should have Homoscedasticity (Constant Variance)
    (同方差性 = when the 'noise' of your model can be described as random) (aka '齐方差检验' )""" # 
# Also known as the assumption for 'constant variance', means that the variability in the response variable
    # is the same at all levels of the explanatory variable
        # In other word, the 'spread' of the residual value does not increase with x (the independent variable)
# Can use scatter plot between 'x/ index of residual' and 'residual' to show whether residual move with 'x/ index'
# Can use BP Test:
    #
###############################################################################
""" # 5. No Autocorrelation between the residuals """ #
# Linear regression requires that the residuals have very little or no auto-correlation in the data.
    # auto-correlation happens when the residuals are not independent of each other, 
        # e.g. error(i+1) is not independent from error(i)
    # Can Use ACF Plot
    # Can Use Durbin-Watson(DW) Test (Rule of Thumb:  1.5 < d < 2.5 )
###############################################################################
""" # 6. No Endogeneity(omitted variable resulting in the error term/ error is also related to independent variable)(无内生性) """
# x(i) is not affecting error (e)
# y(i1) = y(i2) + a0 + a1 * x(i1) + a2 * x(i2) + a3 * x(i3) + ... + ak * x(ik) + error(i)
# In this case, y(i2) is 'endogeneous' when y(i2) is correlated with error(i)
    # y(i2) will be correlaed with error(i) on 3 scenarios:
        # 1. there are omitted variables that are correlated with y(i2) and y(i1) 
            # e.g. omitting innate ability when estimating wage-rate, using education
                # because innate ability is correlated with both wage-rate and eduction
        # 2. y(i2) is measured with errors: Measurement Error
            # e.g. using 'household-income' to represent 'neighbourhood-quality', 
                # whereas not considering 'crime rate', 'pollution'.
                # This means that 'household-income' has to explain more than what it originally affect
                # so that the coefficient of 'household-income' is likely to be exaggerated
        # 3. y(i1) and y(i2) are simultaneously determined (both dependent variable and independent variable affect each other)
            # e.g. under a progressive tax system, tax rate is endogeneous in the model that examines the effect of
                # net-of-tax rate on gross wage income (gross_income = f(tax_rate) )
# Can use Hauseman Test
    # # Note: Hausman Test is valid only whtn 'Homoscedasticity' criteria is met.
#
###############################################################################     
""" Outliers have been kept (consistent with model) or removed (affect model's accuracy) """ # 
#
#
#
""" Normality is assumed for large sample size, based on the Central Limit Theroem """ # or can use Q-Q Plot to test normality
""" Zero mean of the distribution of the error is accomplished by the inclusion of intercept """ # 
""" Independence of observations """ # e.g. no duplication should be included in the model (hierarchical/ nested/ clustered data)
""" Independence of Errors """ # The errors of the response variables are uncorrelated with each other
    # Bayesian Linear Regression is a general way of handling correlated errors.
    #
###############################################################################     
# In[]:
# Y = a * X + b
# Slope is the correlationship between the 2 variables
# Intercept i sthe mean of Y minus the Slope times the mean of X
# "Ordinary Least Square(OLS)" = Least squares minimizes the sum of squared errors
# R-Square = "coefficient of determination" = 1.0 - (Sum of Squared Errors)/ (Sum of Squared Variation from Mean)
import numpy as np
from pylab import scatter #
#
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000) * 3)
scatter(pageSpeeds, purchaseAmount)
#
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount) # Return parameters of the model
#
stats.linregress(pageSpeeds, purchaseAmount) # 
#
r_value ** 2
#
import matplotlib.pyplot as plt
def predict(x):
    return slope * x + intercept
fitLine = predict(pageSpeeds)
#
plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitLine, c='r')
plt.show()
# In[]:
# Import the libraries
# Pandas is high-performance, easy-to-use data structures and data analysis tools for the Python programming language
import pandas as pd
# Numpy is the fundamental package for scientific computing with Python
import numpy as np
# Matplotlib is used for graphs
import matplotlib.pyplot as plt
# seaborn Seaborn is a Python visualization library based on matplotlib. 
    # It provides a high-level interface for drawing attractive statistical graphics
import seaborn as sns
# %matplotlib inline is magic command. This performs the necessary behind-the-scenes setup for IPython to work correctly 
    # hand in hand with matplotlib
%matplotlib inline
# Reading USA_Housing File 
import os
os.chdir('C:\Python_Project\Linear_Regression')
USAhousing = pd.read_csv('USA_Housing.csv')
# To view first 5 rows of dataframe
USAhousing.head()
# To know data type and non null values in a dataframe
USAhousing.info()
# Describe function provide descriptive statistics Output table of data 
USAhousing.describe()
# To visualize corelation between different variables
sns.pairplot(USAhousing)
# To visualize distribution of price
sns.distplot(USAhousing['Price'])
# Another way to visualize corelation between different variables
sns.heatmap(USAhousing.corr())
# Dividing independent variables and target variable
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
# Importing train_test_split() for the purpose of spliting the data into test and train sets
from sklearn.model_selection import train_test_split
# spliting the data into test (40 percent) and train sets (60 percent) with 101 random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
# Importing linear regression function
from sklearn.linear_model import LinearRegression
# Specifying a variable to LinearRegression()
lm = LinearRegression()
# Creating model using train set
lm.fit(X_train,y_train)
# print the intercept
print(lm.intercept_)
# To know coefficients of regression
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
# Predicting test set
predictions = lm.predict(X_test)
# Scatter plot to see our prediction
plt.scatter(y_test,predictions)
# Distribution of Error
sns.distplot((y_test-predictions),bins=50);
# Importing metrics function to see MSE, MAE & RMSE of our prediction
from sklearn import metrics
#
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# In[]: Simple Linear Regression (Ordinary Least Square) 
# min[(actual_ob_1 - estimated_1)^2 + (actual_ob_2 - estimated_2)^2 + (actual_ob_3 - estimated_3)^2 ]^0.5
import os
os.chdir('C:/Python_Learn/Udermy_Course/Simple_Linear_Regression')
# 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
# Splitting the dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#
# Fitting Simple Linear Regression to the Training Set
# 
from sklearn.linear_model import LinearRegression
# Create a model "regressor"
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor
# Predicting the Test set results
y_pred = regressor.predict(X_test)
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary Earned(USD/ Year)')
plt.show()
# 
#plt.scatter(X_train, y_train, color = 'red')
#plt.scatter(X_test, y_pred, color = 'blue')
##plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.title('Salary vs Experience')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary Earned(USD/ Year)')
#plt.show()
#
# Visualising the Test Set Results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary Earned(USD/ Year)')
plt.show()
#
# In[]: 
""" Multivariant Linear Regression """ 
# More than one variable influences the dependent variable
# Assumptions for Linear Regerssion:
# [1] Linearity:
# [2] Homoscedasticity:
# [3] Multivariate Normality:
# [4] Independence of errors:
# [5] Lack of Multicolliearity: 
# Y = B0 + B1*X1 + B2*X2 + B3*X3 + ... + 
# P-Value = The possibility that the scenario we are testing can show up.
#
# [1] "All-in Method" = Prior Knowledge / Have to include those variables / Preparing for a Backward Elimination
# [2] "Backward Elimination" = 
# [3]
# In[]: 
# OLS for MultiVariant
import pandas as pd
df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')
#
import numpy as np
df1 = df[['Mileage', 'Price']]
bins = np.arange(0, 50000, 10000)
groups = df1.groupby(pd.cut(df1['Mileage'], bins)).mean()
print(groups.head())
groups['Price'].plot.line()
#
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
#
X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']
#
X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].as_matrix())
print(X)
#
est = sm.OLS(y, X).fit() # fit the data to the model
est.summary()
#
y.groupby(df.Doors).mean()
#
scaled = scale.transform([[45000, 8, 4]])
print(scaled)
predicted = est.predict(scaled[0])
print(predicted)
# In[]: 
#SST = "Sum of Squares Total"
#SSR = "Sum of Squares Regression"
#SSE = "Sum of Squares Error" = RSS = "Residual/Remaining/Unexplained Sum of Squares"
#SST = SSR + SSE
#OLS = "Ordinary Least Squares" = "Lowest error"
#R-squared = SSR/SST = 1-SSE/SST = "Goodness of Fit"
#
""" Dealing with Catrgorical Data - Dummy Variable """
#
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import os
os.chdir('C:\Python_Project\Linear_Regression\Data')
#
df = pd.read_csv('Dummy_Variable.csv')
df.head()
df.describe(include = 'all')
df['Attendance'].unique()
#
df['Attendance'] = df['Attendance'].map({'No':0, 'Yes':1})
#
y = df['GPA']
x1 = df[['SAT', 'Attendance']]
sns.pairplot(df) # Check Linearity with scatter-plots
import scipy.stats as stats
stats.probplot(df['SAT'], dist = 'norm', plot = plt ) # Use Q-Q plot to test whether the variable follows a normal distribution
stats.probplot(df['Attendance'], dist = 'norm', plot = plt ) # 
df['Attendance'].hist()
df['GPA'].hist()
df['SAT'].hist()
#
#
#pip install researchpy
#import researchpy as rp
#rp.summary_cont(df[['SAT', 'GPA']])
covariance = np.cov(df['GPA'], df['SAT'])
covariance
# 
corr_matrix = df.corr() # Correlation Matrix
sns.heatmap(corr_matrix, annot = True) # Heatmap for Correlation Matrix
#
x = sm.add_constant(x1) # Add constant to test whether the intercept is equal to zero
result = sm.OLS(y, x).fit()
result.summary()
result.wu_hausman()
#
result.params # show parameters, which are coefficients
result.bse # Show standard-errors of each variable
result.rsquared # Show R-Squared value
result.predict() # Predicted values based on the generated regression
#
x2 = sm.add_constant(df['SAT'])
result2 = sm.OLS(y, x2).fit()
result2.summary()
#
plt.scatter(df['SAT'], y)
yhat_no = 0.6439 + 0.0014 * df['SAT'] # Attendance = 0
yhat_yes = 0.8665 + 0.0014 * df['SAT'] # Attendance = 1
yhat = 0.0275 + 0.0017 * df['SAT'] # When not using 'Attendance' as a variable
fig = plt.plot(df['SAT'], yhat_no, lw = 2, c = 'blue', label = 'regression line1')
fig = plt.plot(df['SAT'], yhat_yes, lw = 2, c = 'yellow', label = 'regression line2')
fig = plt.plot(df['SAT'], yhat, lw = 4, c = 'red', label = 'old regression')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
#plt.show() # only needed in Jupyter notebook
#
new_data = pd.DataFrame({'const': 1, 'SAT': [1700, 1670], 'Attendance': [0, 1]}) # Create a New DataFrame to be tested
#new_data = new_data[['const', 'SAT', 'Attendance']]
#
predictions = result.predict(new_data)# apply the New DataFrame to the Prediction Function, which is generated from the Regression
predictions
#
predictionsdf = pd.DataFrame({'Predictions': predictions}) # Generate the DataFrame to be combined to the New DataFrame
joined = new_data.join(predictionsdf, how = 'left')
joined
joined.rename(index = {0: 'Bob', 1 : 'Alice'})



# In[]:
#
corr = df.corr() # Correlation Matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, cmap = 'coolwarm', vmin = -1, vmax = 1)
fig.colorbar(cax)
ticks = np.arange(0, len(df.columns), 1)
ax.set_xticks(ticks)
plt.xticks(rotation = 90)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
#
cluster_map = sns.clustermap(corr, cmap = 'YlGnBu', linewidth = 0.1) #
plt.setp(cluster_map.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)




# In[]: Test for Homoscedasticity
np.random.seed(20)
x = np.arange(20)
y_homo = [x*2 + np.random.rand(1) for x in range(20)]  ## homoscedastic error
y_hetero = [x*2 + np.random.rand(1)*2*x for x in range(20)]  ## heteroscedastic error
#
x_reshape = x.reshape(-1, 1)
#
from sklearn.linear_model import LinearRegression
linear_homo = LinearRegression()
linear_homo.fit(x_reshape, y_homo)
#
linear_hetero = LinearRegression()
linear_hetero.fit(x_reshape, y_hetero)
#



# In[]: Example - Tests for Assumptions
""" # 1. Load Data """
import pandas as pd
from sklearn.datasets import load_boston
# load data
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
X.drop('CHAS', axis=1, inplace=True)
y = pd.Series(boston.target, name='MEDV')
# inspect data
X.head()
""" # 2. Running Linear Regression """
from sklearn.linear_model import LinearRegression
#
lin_reg = LinearRegression()
lin_reg.fit(X, y)
#
print(f'Coefficients: {lin_reg.coef_}')
print(f'Intercept: {lin_reg.intercept_}')
print(f'R^2 score: {lin_reg.score(X, y)}')
# Check Summary Table
import statsmodels.api as sm
X_constant = sm.add_constant(X)
lin_reg = sm.OLS(y,X_constant).fit()
lin_reg.summary()
#
""" Gauss-Markov Theorem: 
        in a linear regression model, the OLS estimator gives the Best-Linear-Unbiased-Estimator(BLUE) of the coefficient, 
            provided that:
            1. The expectation of errors (residuals) is Zero (0)
            2. The errors are uncorrelated
            3. The errors have equal variance - homoscedasticity of errors
"""
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (12.0, 8.0)
# In[]: Step 1: Test Linearity
def linearity_test(model, y):
    '''
    Function for visually inspecting the assumption of linearity in a linear regression model.
    It plots observed vs. predicted values and residuals vs. predicted values.
    
    Args:
    * model - fitted OLS model from statsmodels
    * y - observed values
    '''
    fitted_vals = model.predict()
    resids = model.resid

    fig, ax = plt.subplots(1,2)
    
    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='Observed')

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
    
linearity_test(lin_reg, y)    
# 
""" 1. Check Whether the Expectation (mean) of Residual = 0 """
lin_reg.resid.mean()
lin_reg.resid.hist() # 
lin_reg.resid.mean() < 0.01
plt.scatter(lin_reg.resid.index, lin_reg.resid.values) # 
plt.scatter(X['INDUS'], lin_reg.resid.values) # 
plt.scatter(X['NOX'], lin_reg.resid.values) # 
#
""" 2. VIF
    # If no features are correlated, then all values for VIF will = 1.0  
        # a general rule-of-thump is that VIF should be less than """
from statsmodels.stats.outliers_influence import variance_inflation_factor
#
vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=X.columns).T
# In[]: Test for Homoscedasticity
""" 3. Homoscedasticity of Errors """
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)
#
def homoscedasticity_test(model):
    '''
    Function for testing the homoscedasticity of residuals in a linear regression model.
    It plots residuals and standardized residuals vs. fitted values and runs Breusch-Pagan and Goldfeld-Quandt tests.
    
    Args:
    * model - fitted OLS model from statsmodels
    '''
    fitted_vals = model.predict()
    resids = model.resid
    resids_standardized = model.get_influence().resid_studentized_internal

    fig, ax = plt.subplots(1,2)

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Residuals vs Fitted', fontsize=16)
    ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

    sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Scale-Location', fontsize=16)
    ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')

    bp_test = pd.DataFrame(sms.het_breuschpagan(resids, model.model.exog), 
                           columns=['value'],
                           index=['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'])

    gq_test = pd.DataFrame(sms.het_goldfeldquandt(resids, model.model.exog)[:-1],
                           columns=['value'],
                           index=['F statistic', 'p-value'])

    print('\n Breusch-Pagan test ----')
    print(bp_test)
    print('\n Goldfeld-Quandt test ----')
    print(gq_test)
    print('\n Residuals plots ----')
#
homoscedasticity_test(lin_reg)
# In[]: Test for Auto-correlation between the Residuals (ACF)
""" No Autocorrelation of Residuals """
# This assumption is especially dangerous in time-series models.
import statsmodels.tsa.api as smt
#
acf = smt.graphics.plot_acf(lin_reg.resid, lags=40 , alpha=0.05) #
acf.show()
# In[]: Test thr normality of the Residual
from scipy.stats.stats import pearsonr

for column in X.columns:
    corr_test = pearsonr(X[column], lin_reg.resid)
    print(f'Variable: {column} --- correlation: {corr_test[0]:.4f}, p-value: {corr_test[1]:.4f}')
#
""" Normality of Residual """
from scipy import stats

def normality_of_residuals_test(model):
    '''
    Function for drawing the normal QQ-plot of the residuals and running 4 statistical tests to 
    investigate the normality of residuals.
    
    Arg:
    * model - fitted OLS models from statsmodels
    '''
    sm.ProbPlot(model.resid).qqplot(line='s');
    plt.title('Q-Q plot');
#
    jb = stats.jarque_bera(model.resid)
    sw = stats.shapiro(model.resid)
    ad = stats.anderson(model.resid, dist='norm')
    ks = stats.kstest(model.resid, 'norm')
#
    print(f'Jarque-Bera test ---- statistic: {jb[0]:.4f}, p-value: {jb[1]}')
    print(f'Shapiro-Wilk test ---- statistic: {sw[0]:.4f}, p-value: {sw[1]:.4f}')
    print(f'Kolmogorov-Smirnov test ---- statistic: {ks.statistic:.4f}, p-value: {ks.pvalue:.4f}')
    print(f'Anderson-Darling test ---- statistic: {ad.statistic:.4f}, 5% critical value: {ad.critical_values[2]:.4f}')
    print('If the returned AD statistic is larger than the critical value, then for the 5% significance level, \
          the null hypothesis that the data come from the Normal distribution should be rejected. ')
#
normality_of_residuals_test(lin_reg)
#
# In[]:
from statsmodels.sandbox.regression.gmm import IV2SLS
#from __future__ import division
#
#%matplotlib inline
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 18.5, 10.5
#
def print_resids(preds, resids):
    ax = sns.regplot(preds, resids);
    ax.set(xlabel = 'Predicted values', ylabel = 'errors', title = 'Predicted values vs. Errors')
#    plt.show();
#
#
import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
#
fertility = pd.read_stata("http://rlhick.people.wm.edu/econ407/data/fertility.dta")
fertility.shape
fertility.plot.scatter('educ', 'agefbrth'); fertility.plot.scatter( 'agefbrth', 'children');
#
print(fertility[fertility['usemeth'] == 1]['agefbrth'].dropna().describe())
sns.distplot(fertility[fertility['usemeth'] == 1]['agefbrth'].dropna() );
#
print(fertility[fertility['usemeth'] == 0]['agefbrth'].dropna().describe())
sns.distplot(fertility[fertility['usemeth'] == 0]['agefbrth'].dropna() );
#
no_null = fertility[(fertility['agefbrth'].notnull()) & (fertility['educ'].notnull()) & 
                    (fertility.monthfm.notnull()) & (fertility['ceb'].notnull()) & (fertility['idlnchld'].notnull())] 
#
print("lost {} samples of data out of a total {} samples".format(fertility.shape[0] - no_null.shape[0],
                                                                 fertility.shape[0] ))
#
ind_vars = ['monthfm', 'ceb', 'educ', 'idlnchld']
dep_var = 'agefbrth'
x = no_null[ind_vars] 
y = no_null[dep_var]
#
x_const = sm.add_constant(x)
#
first_model_results = sm.OLS(y, x_const, missing = 'drop').fit()
#
#results = first_model.fit()
#
first_model_results.summary() # 
#
print( "the descriptive statistics for the errors and a histogram of them:\n\n", first_model_results.resid.describe())
sns.distplot(first_model_results.resid);
# Part 3: Pick an instrument and test for relevancy and strength
rel = ['monthfm', 'ceb', 'electric']
endog = 'educ'

dropped_na = fertility[(fertility.monthfm.notnull()) & (fertility.ceb.notnull()) & (fertility.electric.notnull())
                    & (fertility.educ.notnull())]

only_exog = sm.add_constant(dropped_na[rel])
relevancy_results = sm.OLS(dropped_na[endog], only_exog).fit()

relevancy_results.summary()
# Run the hypothesis test that the coefficient on electric is 0:
hypothesis = '(electric = 0)'
print(relevancy_results.f_test(hypothesis))
# Part 4: Instrumenting using two-stage least squares
#
no_null_iv = fertility[(fertility['agefbrth'].notnull()) & (fertility['electric'].notnull()) & 
                    (fertility['monthfm'].notnull()) & (fertility['ceb'].notnull()) & (fertility['educ'].notnull())
                      & (fertility['idlnchld'].notnull())]
endog = no_null_iv['agefbrth']
exog = no_null_iv[['monthfm', 'ceb', 'idlnchld', 'educ']]
instr = no_null_iv[['monthfm', 'ceb', 'idlnchld', 'electric']]
dep_var_iv = no_null_iv['agefbrth']
#
exog_constant = sm.add_constant(exog)
instr_constant = sm.add_constant(instr)
no_endog_results = IV2SLS(endog, exog_constant, instrument = instr_constant).fit()
#
no_endog_results.summary()
#
print_resids(no_endog_results.predict(), no_endog_results.resid)
#
print("the descriptive statistics for the errors and a histogram of them:\n\n", no_endog_results.resid.describe())
sns.distplot(no_endog_results.resid);
# Part 5: replicate using matrix algebra
x_mat_ols = np.matrix(x_const)
y_mat_ols = np.matrix(y)
y_mat_ols = np.reshape(y_mat_ols, (-1, 1)) #reshape so that its a single column vector, not row vector
b_ols = np.linalg.inv(x_mat_ols.T*x_mat_ols)*x_mat_ols.T*y_mat_ols
print(b_ols)
#
y_iv_mat = np.matrix(endog)
y_iv_mat = np.reshape(y_iv_mat, (-1, 1))
z_mat = np.matrix(instr_constant)
x_mat_iv = np.matrix(exog_constant) 
np.linalg.inv(z_mat.T * x_mat_iv)*z_mat.T*y_iv_mat
# Part 6: Hausman-Wu test for endogeneity
# add relevancy equation residuals on to the endogenous matrix
x_const['relevancy_resids'] = relevancy_results.resid

# run endogenous regression now with residuals added in
endog_test_results = sm.OLS(y, x_const, missing = 'drop').fit()

endog_test_results.summary()
#
null_hypothesis = '(relevancy_resids = 0)'
print(endog_test_results.f_test(null_hypothesis))
# We reject the null hypothesis that education is exogenous and conclude that education is indeed an endogenous variable.
# Part 7: Add another instrument
two_ivs = fertility[(fertility['agefbrth'].notnull()) & (fertility['electric'].notnull()) & 
                    (fertility['monthfm'].notnull()) & (fertility['ceb'].notnull()) & (fertility['educ'].notnull())
                      & (fertility['idlnchld'].notnull()) & (fertility['urban'].notnull())]

endog = two_ivs['agefbrth']
exog = two_ivs[['monthfm', 'ceb', 'idlnchld', 'educ']]
instr = two_ivs[['monthfm', 'ceb', 'idlnchld', 'electric', 'urban']]


exog_constant = sm.add_constant(exog)
instr_constant = sm.add_constant(instr)
two_iv_results = IV2SLS(endog, exog_constant, instrument = instr_constant).fit()
#
two_iv_results.summary()
#
print_resids(two_iv_results.predict(), two_iv_results.resid)
#

print(two_iv_results.resid.describe())
sns.distplot(two_iv_results.resid);
#
rel = ['monthfm', 'ceb', 'electric', 'urban']
endog = 'educ'

only_exog = sm.add_constant(fertility[rel])
relevancy_results = sm.OLS(fertility[endog], only_exog, missing = 'drop').fit()

relevancy_results.summary()
#
null_hypotheses = '(electric = 0), (urban = 0)'
print(relevancy_results.f_test(null_hypotheses))
#
# In[]: 2-stage Regression = one method to resolve 'Endogeneity'
# Import and select the data
import pandas as pd
import statsmodels.api as sm
#
df4 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable4.dta')
df4 = df4[df4['baseco'] == 1]
# Add a constant variable
df4['const'] = 1
# Fit the first stage regression and print summary
results_fs = sm.OLS(df4['avexpr'], 
                    df4[['const', 'logem4']],
                    missing='drop').fit()
print(results_fs.summary()) # Result of 1st stage regression
df4['predicted_avexpr'] = results_fs.predict() 
results_ss = sm.OLS(df4['logpgp95'], 
                    df4[['const', 'predicted_avexpr']]).fit() # Second Stage of Regression
print(results_ss.summary()) # the new coefficient is supposed to be 'unbiased'
# 2SLS Regression
from linearmodels.iv import IV2SLS
iv = IV2SLS(dependent=df4['logpgp95'],
            exog=df4['const'],
            endog=df4['avexpr'],
            instruments=df4['logem4']).fit(cov_type='unadjusted')

print(iv.summary)
# In[]: 
# Load in data
df4 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable4.dta')
# Add a constant term
df4['const'] = 1
# Estimate the first stage regression
reg1 = sm.OLS(endog=df4['avexpr'],
              exog=df4[['const', 'logem4']],
              missing='drop').fit()
# Retrieve the residuals
df4['resid'] = reg1.resid
# Estimate the second stage residuals
reg2 = sm.OLS(endog=df4['logpgp95'],
              exog=df4[['const', 'avexpr', 'resid']],
              missing='drop').fit()
print(reg2.summary())
#
#
# In[]:
#
import statsmodels.api as sm
from sklearn import linear_model
#
x1 = [26.0, 31.0, 47.0, 51.0, 50.0, 49.0, 37.0, 33.0, 49.0, 54.0, 31.0, 49.0, 48.0, 49.0, 49.0, 47.0, 44.0, 48.0, 35.0, 43.0]
y1 = [116.0, 94.0, 100.0, 102.0, 116.0, 116.0, 68.0, 118.0, 91.0, 104.0, 78.0, 116.0, 90.0, 109.0, 116.0, 118.0, 108.0, 119.0, 110.0, 102.0]
#
# Fit and summarize statsmodel OLS model
x1 = sm.add_constant(x1)
model_sm = sm.OLS(y1, x1)
result_sm = model_sm.fit()
result_sm.summary()
#
# Create sklearn linear regression object
ols_sk = linear_model.LinearRegression(fit_intercept=True)
# fit model
model_sk = ols_sk.fit(pd.DataFrame(x1), pd.DataFrame(y1))
# sklearn coefficient of determination
coefofdet = model_sk.score(pd.DataFrame(x1), pd.DataFrame(y1))
#
print('sklearn R^2: ' + str(coefofdet))
model_sk.coef_
model_sk.intercept_
