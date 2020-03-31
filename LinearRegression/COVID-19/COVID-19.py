#!/usr/bin/env python
# coding: utf-8

# # COVID-19 - Global Cases - EDA  and Forecasting

# This is the data repository for the 2019 Novel Coronavirus Visual Dashboard operated by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE). Also, Supported by ESRI Living Atlas Team and the Johns Hopkins University Applied Physics Lab (JHU APL).
# 
# Data is sourced from https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data
# 
# 
# * Visual Dashboard (desktop):
# https://www.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
# 
# * Visual Dashboard (mobile):
# http://www.arcgis.com/apps/opsdashboard/index.html#/85320e2ea5424dfaaa75ae62e5c06e61
# 
# * Lancet Article:
# An interactive web-based dashboard to track COVID-19 in real time
# 
# * Provided by Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE):
# https://systems.jhu.edu/
# 
# * Data Sources:
# 
#     - World Health Organization (WHO): https://www.who.int/
#     - DXY.cn. Pneumonia. 2020. http://3g.dxy.cn/newh5/view/pneumonia.
#     - BNO News: https://bnonews.com/index.php/2020/02/the-latest-coronavirus-cases/
#     - National Health Commission of the People’s Republic of China (NHC):
# http://www.nhc.gov.cn/xcs/yqtb/list_gzbd.shtml
#     - China CDC (CCDC): http://weekly.chinacdc.cn/news/TrackingtheEpidemic.htm
#     - Hong Kong Department of Health: https://www.chp.gov.hk/en/features/102465.html
#     - Macau Government: https://www.ssm.gov.mo/portal/
#     - Taiwan CDC: https://sites.google.com/cdc.gov.tw/2019ncov/taiwan?authuser=0
#     - US CDC: https://www.cdc.gov/coronavirus/2019-ncov/index.html
#     - Government of Canada: https://www.canada.ca/en/public-health/services/diseases/coronavirus.html
#     - Australia Government Department of Health: https://www.health.gov.au/news/coronavirus-update-at-a-glance
#     - European Centre for Disease Prevention and Control (ECDC): https://www.ecdc.europa.eu/en/geographical-distribution-2019-ncov-cases
#     - Ministry of Health Singapore (MOH): https://www.moh.gov.sg/covid-19
#     - Italy Ministry of Health: http://www.salute.gov.it/nuovocoronavirus
# 
#     - Additional Information about the Visual Dashboard:
# https://systems.jhu.edu/research/public-health/ncov/
# 
# Contact Us:
# 
# Email: jhusystems@gmail.com
# 
# Terms of Use:
# 
# This GitHub repo and its contents herein, including all data, mapping, and analysis, copyright 2020 Johns Hopkins University, all rights reserved, is provided to the public strictly for educational and academic research purposes. The Website relies upon publicly available data from multiple sources, that do not always agree. The Johns Hopkins University hereby disclaims any and all representations and warranties with respect to the Website, including accuracy, fitness for use, and merchantability. Reliance on the Website for medical guidance or use of the Website in commerce is strictly prohibited.

# __For better viewing experience, I recommend to enable NBextensions as guided @__
# 
# https://github.com/lsunku/DataScience/tree/master/JupyterNotebook

# # Steps invoved in this notebook

# 1. Import Python Libraries for data analysis and ML 
# 2. Local user defined functions
# 3. Sourcing the Data
# 4. Inspect and Clean the Data
# 5. Exploratory Data Analysis
# 6. Preparing the data for modelling(train-test split, rescaling etc)
# 7. Model evaluation for Advanced Regression Criteria
# 8. Linear Regression Model for World Wide Case Predictions
# 9. Linear Regression Model for Italy Predictions
# 10. Linear Regression Model for US Predictions
# 11. Linear Regression Model for Spain Predictions
# 12. Linear Regression Model for Germany Predictions
# 13. Linear Regression Model for India Predictions

# __Notes:__ Currently, I have used only time_series_covid19_confirmed_global for the following analysis. When I get time, I shall enhance the same with additional files time_series_covid19_deaths_global, time_series_covid19_recovered_global and integrate with daily reports.

# # __Import Python Functions__

# In[284]:


# Local classes and Local flags

# Local Classes
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
    
# Debug flag for investigative purpose
DEBUG = 0

# Default random_state
rndm_stat = 42


# In[285]:


# Python libraries for Data processing and analysis
import time as time
strt = time.time()
import pandas as pd
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)
pd.options.mode.use_inf_as_na = True
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import glob
from matplotlib.pyplot import figure
import warnings
import math
import itertools
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
from math import sqrt
import re
from prettytable import PrettyTable

# ML Libraries
import statsmodels
import statsmodels.api as sm
import sklearn as sk
from sklearn.model_selection import train_test_split,GridSearchCV, KFold,RandomizedSearchCV,StratifiedKFold
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler,OrdinalEncoder,LabelEncoder,Normalizer,RobustScaler,PowerTransformer,PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor


# # __Local User Defined Functions__

# ## Local functions for data overview and data cleaning

# In[286]:


# local functions

# Function to read a file & Store it in Pandas
# read_file takes either csv or excel file as input and reuturns a pandas DF and
# also prints head, tail, description, info and shape of the DF
def read_file(l_fname,l_path,head=0):
    i = l_fname.split(".")
    f_path = l_path+'/'+l_fname
    print(f_path,i[0],i[1])
    if (i[1] == "xlsx"):
        l_df = pd.read_excel(f_path,header=head,encoding = "ISO-8859-1",infer_datetime_format=True)
    elif (i[1] == "csv"):
        l_df = pd.read_csv(f_path,header=head,encoding = "ISO-8859-1",infer_datetime_format=True)
    ov_df(l_df)
    return(l_df)

# Function to get the Overview of DataFrame
# take df as input and prints head, tail, description, info and shape of the DF
def ov_df(l_df):
    print(color.BOLD+color.PURPLE + 'Inspect and Explore the Dataset' + color.END)
    print("\n#####################  DataFrame Head  ######################")
    print(l_df.head(3))
    print("\n#####################  DataFrame Tail  ######################")
    print(l_df.tail(3))
    print("\n#####################  DataFrame Info  ######################")
    print(l_df.info())
    print("\n####################  DataFrame Columns  ####################")
    print(list(l_df.columns))
    print("\n####################  DataFrame Shape  ####################")
    print("No of Rows",l_df.shape[0])
    print("No of Columns",l_df.shape[1])

# Function per_col_null takes a df as input and prints summary of Null Values across Columns
def per_col_null(l_df):
    print("\n############  Missing Values of Columns in %  ############")
    col_null = round((l_df.isnull().sum().sort_values(ascending=False)/len(l_df))*100,4)
    print(col_null[col_null > 0])  


# # __Sourcing the Data__

# ## Read the train.csv

# In[287]:


# Set the path and file name
folder=r"C:\My Folders\OneDrive\Surface\Sadguru\Lakshmi\Study\IIIB_PGDS\Hackathon\COVID_19\COVID-19\csse_covid_19_data\csse_covid_19_time_series"
file="time_series_covid19_confirmed_global.csv"

# Read file using local functions. read_file takes either csv or excel file as input and reuturns a pandas DF and
# also prints head, tail, description, info and shape of the DF
raw_df = read_file(file,folder)


# In[288]:


# transpose and format the columns
raw_df = raw_df.drop(["Province/State","Lat","Long"],axis=1).set_index("Country/Region").T.reset_index().rename(columns={'index':'Date'}).rename_axis("",axis="columns")


# In[289]:


ov_df(raw_df)


# ## Inspect the Column Data Types of c_df

# In[290]:


# Analyze Categorical, Numerical and Date variables of Application Data
print(color.BOLD+"Categorical and Numerical Variables"+ color.END)
display(raw_df.dtypes.value_counts())
print(color.BOLD+"Numerical Integer Variables"+ color.END)
display(raw_df.select_dtypes(include='int64').dtypes)
print(color.BOLD+"Categorical Variables"+ color.END)
display(raw_df.select_dtypes(include=object).dtypes)
print(color.BOLD+"Numerical Float Variables"+ color.END)
display(raw_df.select_dtypes(include='float64').dtypes)


# In[291]:


# Change the Date format
raw_df["Date"] = pd.to_datetime(raw_df["Date"],infer_datetime_format=True)


# In[292]:


# as the given data is segrated in some countries which are epicenters and for some, it is not. To make it uniform, I sum up the data across countries

dt = raw_df.pop("Date")
dt.head()


# In[293]:


# Aggregate the data across columns as there are columns with same column name
c_df = raw_df.groupby(by=raw_df.columns,axis=1).agg(sum)

c_df.head()


# In[294]:


c_df.insert(0,"Date",dt)
c_df.head()


# # __Exploratory Data Analysis__

# ## Inspect the Null Values in c_df

# In[295]:


# Null values in the Application DF. 
# per_col_null is local function which returns the % of null columns which are non zero
per_col_null(c_df)


# ## Derived Columns

# In[296]:


c_df["WW"] = c_df.sum(axis=1)

c_df.head()


# In[297]:


import plotly.express as ply
import plotly.graph_objects as go
import cufflinks as cf


# In[298]:


cntry_li = list(c_df.columns)
cntry_li.remove("Date")


# In[299]:


fig = go.Figure()
for i in cntry_li:
    fig.add_trace(go.Scatter(x=c_df["Date"],y=c_df[i],mode='lines+markers',name=i))
fig.update_layout(
    margin=dict(l=30, r=20, t=25, b=25),

)  
#fig.update_layout(yaxis_type="log")
fig.show()


# ## List of countries which are contributing to high number of positive cases

# In[300]:


hi_co_li = [i for i,j in (c_df[cntry_li].iloc[-1] > 1500).items() if j == True]
print(hi_co_li)


# In[301]:


fig = go.Figure()
for i in hi_co_li:
    fig.add_trace(go.Scatter(x=c_df["Date"],y=c_df[i],mode='lines+markers',name=i))
fig.update_layout(
    margin=dict(l=40, r=30, t=25, b=25),

)  
#fig.update_layout(yaxis_type="log")
fig.show()


# ## Analyze Categorical Columns of the c_df

# In[302]:


c_df.insert(0,"Day",np.arange(1,len(c_df)+1))


# In[303]:


c_df.head()


# In[304]:


# Create a list of numerical and categorical variables for future analysis
c_num_li = list(c_df.select_dtypes(include=np.number).columns)
c_cat_li = list(c_df.select_dtypes(exclude=np.number).columns)
print(color.BOLD+"\nNumerical Columns -"+color.END,c_num_li)
print(color.BOLD+"\nCategorical Columns -"+color.END,c_cat_li)


# ## Analyze Numerical Columns of the c_df

# In[305]:


# Inspect the Categorical columns
c_df[c_cat_li].head()


# In[306]:


# Inspect the Numerical columns
c_df[c_num_li].head()


# ## Univariate analysis

# Univariate analysis is performed only on specific countries which are suffering with high number of positive cases

# ### Univariate analysis of Countries which are sufferring with high number of corona cases

# In[307]:


# Inspect list of categorical variables
print(hi_co_li) 


# In[308]:


# Function to plot 2 or more line plots or time series plots
# line_pltly takes a df, dependent variable and variable list of columns
# to plot multiple reg plots
def line_pltly (l_df,l_dep,*args):
    for i in args:
        fig = go.Figure()
        for l in ["WW","China","Korea, South"]:
            fig.add_trace(go.Scatter(x=l_df[l_dep],y=l_df[l],mode='lines+markers',name=l))
        fig.add_trace(go.Scatter(x=l_df[l_dep],y=l_df[i],mode='lines+markers',name=i))
        fig.update_layout(width=800,height=400,hovermode="closest",clickmode="event+select")
        fig.show()


# In[309]:


# dist_plt is local function which takes a df, rows, columns of subplot and name of columns as an argument and 
# plots distribution plots
# Part-1
line_pltly(c_df,"Day",*hi_co_li[0:4])


# In[310]:


# dist_plt is local function which takes a df, rows, columns of subplot and name of columns as an argument and 
# plots distribution plots
# Part-1
line_pltly(c_df,"Day",*hi_co_li[4:8])


# In[311]:


# dist_plt is local function which takes a df, rows, columns of subplot and name of columns as an argument and 
# plots distribution plots
# Part-1
line_pltly(c_df,"Day",*hi_co_li[8:12])


# In[312]:


# dist_plt is local function which takes a df, rows, columns of subplot and name of columns as an argument and 
# plots distribution plots
# Part-1
line_pltly(c_df,"Day",*hi_co_li[12:16])


# In[313]:


# dist_plt is local function which takes a df, rows, columns of subplot and name of columns as an argument and 
# plots distribution plots
# Part-1
line_pltly(c_df,"Day",*hi_co_li[16:20])


# In[314]:


# dist_plt is local function which takes a df, rows, columns of subplot and name of columns as an argument and 
# plots distribution plots
# Part-1
line_pltly(c_df,"Day",*hi_co_li[20:24])


# ## Preparing the data for modelling(encoding,train-test split, rescaling etc)

# In[315]:


# split the data for training and testing
df_train,df_test = train_test_split(c_df,train_size=0.93,random_state=rndm_stat,shuffle=False)
print(df_train.shape)
print(df_test.shape)


# In[316]:


# Extract the serial number and store it for future purposes
trn_date = df_train.pop('Date')
tst_date = df_test.pop('Date')


# In[317]:


print(df_train.head())
print(df_test.head())


# #### Scaling of Test Data LR Model 1 and Model 2 using Standardization

# # __Model Evaluation Criteria__

# ### Model Evaluation Criteria

# Following criteria should be fulfilled for the best model and each model is evaluated based on the following conditions.
# 1. Residuals (Actual Test data and Predicted Test data) should be normally distributed with mean zero.
# 2. Residuals (Actual Test data and Predicted Test data) are independent of each other.
# 3. Residuals (Actual Test data and Predicted Test data)  have constant variance.
# 4. Model should not be overfit.
# 5. Adjusted R-Square should be little less but comapritively closer to R-Square.
# 6. R-Square should be comparitvely high suggesting a good fit.
# 7. R-Square of Test and Train should be closer to each other suggesting that model has worked well with unseen data.
# 8. Check the RMSE, MSE and MAE of each model and compare it among the 3 models.

# # __LR Model using Linear Regression for World Wide Cases__

# __Ridge Regression Steps__
# * 1) Prepare the data for modelling
# * 2) Hyperparameter tuning and selection using GridSearchCV
# * 3) Build the Ridge Regression Model using optimal Lambda value
# * 4) Predict on Train Set
# * 5) Predict on Test Set

# ## Prepare the data for Modelling

# In[318]:


# Prepare the strings to be used
cntry = "WW"
cntry_act = cntry+"_Actuals"
cntry_pred_m1 = cntry+"_Pred_M1"
cntry_pred_m2 = cntry+"_Pred_M2"


# In[319]:


# 2 Models are created and hence 2 copies of df_train and test to perform the analysis
y_train = df_train[cntry].copy(deep=True)
X_train = df_train[["Day"]].copy(deep=True)

y_test = df_test[cntry].copy(deep=True)
X_test = df_test[["Day"]].copy(deep=True)


# In[320]:


# Target variable is removed from predictor variables
display(y_train.head())
display(X_train.head())


# ## Build the LR Model on Training Set

# ### Parameter Tuning and Selection of Degree

# In[321]:


# function to populate linear regression model metrics
def lm_metrics(y_act,y_pred):
    # calculate the RSquared and RMSE for test data and Predicted data
    rsqr = r2_score(y_true=y_act,y_pred=y_pred)
    mar = mean_absolute_error(y_true=y_act,y_pred=y_pred)
    mse = mean_squared_error(y_act, y_pred)
    rmse = sqrt(mean_squared_error(y_act, y_pred))
    return (rsqr,mar,mse,rmse)


# In[322]:


# function to populate evaluation metrics for different degree
def eval_reg (X_trn,y_trn,deg):
    # list of degrees to tune
    deg_li = list(np.arange(2,deg))
    metric_cols = ["Degree","RSquare","MAE","MSE","RMSE"]
    lm_metrics_df = pd.DataFrame(columns = metric_cols)
    
    # regression model
    reg = Lasso(random_state=rndm_stat)
    
    for count, degree in enumerate(deg_li):
        lm = make_pipeline(PolynomialFeatures(degree=degree), reg)
        lm.fit(X_trn, y_trn)
        y_trn_pred = lm.predict(X_trn)
        rsqr,mar,mse,rmse = lm_metrics(y_trn,y_trn_pred)
        lm_metrics_df.loc[count] = [degree,rsqr,mar,mse,rmse]
    display(lm_metrics_df)


# In[323]:


# Populate the results for different degrees
deg = 12
eval_reg(X_train,y_train,12)


# ### Build the Model using the selected degree

# In[324]:


# Build the model with optimal degree.
degree = 8

reg = Lasso(random_state=rndm_stat)
# create an instance using the optimal degree
lm = make_pipeline(PolynomialFeatures(degree), reg)

# fit the model using training data
lm.fit(X_train, y_train)


# ## Predictions on the train set

# In[325]:


# predict using train data
y_train_pred = lm.predict(X_train)


# ### Residual Analysis and validating the assumptions on Train Set

# #### Error terms are normally distributed with mean zero

# In[326]:


# Calculate the Residuals and check if they are normally distributed or not
res_m1 = y_train - y_train_pred
plt.figure(1,figsize=(8,4))
sns.set(style="whitegrid",font_scale=1.2)
sns.distplot(round(res_m1,2),bins=8,color="green")
plt.vlines(round(res_m1,2).mean(),ymin=0,ymax=2,linewidth=3.0,color="black",linestyles='dotted')
plt.title('Distribution of Residual plot Actual and Predicted Train Data')
plt.show()


# In[327]:


# Mean of Residuals
round(res_m1,2).mean()


# * The mean of residuals is observed to be very close 0

# #### Error terms are independent of each other:

# In[328]:


# check if the Residuals are normally distributed or not
plt.figure(1,figsize=(6,4))
sns.set(style="whitegrid",font_scale=1.2)
ax = sns.lineplot(data=res_m1, color="green", label="line")
plt.title('Distribution of Residuals of Train Data')
plt.show()


# * There is no specific visible pattern

# #### Error terms have constant variance (homoscedasticity):

# In[329]:


plt.figure(2,figsize=(6,6))
sns.set(style="whitegrid",font_scale=1.2)
ax1 = sns.regplot(x=y_train,y=y_train_pred,color='green')
plt.title('Linear Regression Plot of Train and Train Pred',fontsize=12)
plt.show()


# * Error terms have constant variance but in the end couple of points are out of the variance

# In[330]:


# calculate the RSquared and RMSE for test data and Predicted data
print(color.BOLD+"\nModel Evalutation metrics of train set with degree ",degree)
rsqr,mar,mse,rmse = lm_metrics(y_train,y_train_pred)

print(color.BOLD+"RSquare of the Model is ",round(rsqr,2))
print(color.BOLD+"Mean Absolute Error of the Model is",round(mar,2))
print(color.BOLD+"MSE of the model is ",round(mse,2))
print(color.BOLD+"RMSE of the model is ",round(rmse,2))


# ### __Observations on Training Set__
# 1. Residuals (Actual Train data and Predicted Train data) are be normally distributed with mean zero.
#    - Here it is close to 0
# 2. Residuals (Actual Train data and Predicted Train data) are independent of each other.
# 3. Residuals (Actual Train data and Predicted Train data)  have constant variance.
# 4. Adjusted R-Square and R-Square are close to each other and Adjusted R-Square is below R-Square.
# ___Hence the basic checks are good on training data, this model can be used on test set for further evaluations___

# ## Prediction and Evaluation on the Test Set

# * Make predictions on the test set (y_test_pred)
# * evaluate the model, r-squared on the test set

# ### Preprocessing of Test Set Data based on Train Set

# In[331]:


display(y_test.head())
display(X_test.head())


# ### Predict on Test Data

# In[332]:


# predict y_test_pred based on our model
y_test_pred = lm.predict(X_test)


# In[333]:


y_test_pred


# ### Model Evalution of Metrics of Test Data

# In[334]:


# calculate the RSquared and RMSE for test data and Predicted data
print(color.BOLD+"\nModel Evalutation metrics of test set with degree ",degree)
rsqr,mar,mse,rmse = lm_metrics(y_test,y_test_pred)

print(color.BOLD+"RSquare of the Model is ",round(rsqr,2))
print(color.BOLD+"Mean Absolute Error of the Model is",round(mar,2))
print(color.BOLD+"MSE of the model is ",round(mse,2))
print(color.BOLD+"RMSE of the model is ",round(rmse,2))
print(color.BOLD+"\nModel Evalutation metrics of test set with degree ",degree)


# ### Residual Analysis and validating the assumptions on Test Set

# #### Error terms are normally distributed with mean zero

# In[335]:


# Calculate the Residuals and check if they are normally distributed or not
res_test_m1 = y_test - y_test_pred
plt.figure(1,figsize=(8,4))
sns.set(style="whitegrid",font_scale=1.2)
sns.distplot(round(res_test_m1,2),bins=10,color="firebrick")
plt.vlines(round(res_test_m1,2).mean(),ymin=0,ymax=2,linewidth=3.0,color="black",linestyles='dotted')
plt.title('Distribution of Residual plot Actual and Predicted Test Data')
plt.show()


# In[336]:


# Mean of Residuals
round(res_test_m1,2).mean()


# * The mean of residuals is observed to be very close 0

# #### Error terms are independent of each other:

# In[337]:


plt.figure(1,figsize=(6,4))
sns.set(style="whitegrid",font_scale=1.2)
ax = sns.lineplot(data=res_test_m1, color="firebrick", label="line")
plt.title('Distribution of Residuals of Test Data')
plt.show()


# * There is no specific visible pattern

# #### Error terms have constant variance (homoscedasticity):

# In[338]:


plt.figure(2,figsize=(6,6))
sns.set(style="whitegrid",font_scale=1.2)
ax1 = sns.regplot(x=y_test,y=y_test_pred,color="firebrick")
plt.title('Linear Regression Plot of Test and Test_Pred',fontsize=12)
plt.show()


# * Error terms have constant variance but in the end couple of points are out of the variance

# #### Distribution of Actual Test Data and Predicted Test Data

# In[339]:


# Plot the distribution of Actual values of Price and Predicted values of Price
plt.figure(1,figsize=(10,4))
sns.set(style="whitegrid",font_scale=1)
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Values of COVID Cases")
sns.distplot(y_test_pred, hist=False, color="b", label="Predicted Values of COVID Cases" , ax=ax1)
sns.distplot((y_test_pred+rmse), hist=False, color="y", label="Predicated Values of Price + RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
sns.distplot((y_test_pred-rmse), hist=False, color="y", label="Predicated Values of Price - RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
plt.title('LR Model I - Distribution of Actual Values of COVID Cases and Predicted Values of COVID Cases',fontsize=12)
plt.show()


# ### Predict on Actual Test Data

# In[340]:


# generate days up to 72. 
X_act_test = np.arange(1,72).reshape(-1,1)


# In[341]:


# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)


# In[342]:


# create a df with predicted values
covid_df = pd.DataFrame()


# In[343]:


# Create a column with Dates and Day. Starting date is 2020-01-22
covid_df["Day"] = np.arange(1,72)
covid_df["Date"] =  pd.date_range(start=c_df.Date[0], end=c_df.Date[0]+pd.to_timedelta(pd.np.ceil(70), unit="D"))
covid_df[cntry_pred_m1] = np.rint(y_act_test_pred)


# ### Model 2 with different polynomial

# In[344]:


# Build the model with optimal degree.
degree = 9
reg = Lasso(random_state=rndm_stat)
lm = make_pipeline(PolynomialFeatures(degree), reg)
lm.fit(X_train, y_train)
# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)
# add it to df
covid_df[cntry_pred_m2] = np.rint(y_act_test_pred)


# In[345]:


# add the data to final df
covid_df[cntry] = c_df[cntry]


# In[346]:


covid_df[["Date",cntry_pred_m1,cntry_pred_m2,cntry]].tail(10)


# ## __Plot the Actual vs Predicted Models__

# In[347]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m1],mode='lines+markers',name=cntry_pred_m1))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m2],mode='lines+markers',name=cntry_pred_m2))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=c_df[cntry],mode='lines+markers',name=cntry_act))
fig.update_layout(margin=dict(l=40, r=30, t=25, b=25),) 
fig.update_xaxes(nticks=30)
fig.show()


# In[348]:


covid_df[cntry] = c_df[cntry]


# In[349]:


covid_df.tail(10)


# # __LR Model using Linear Regression for - Italy__

# ## Explore the data for Modelling

# In[350]:


# Prepare the strings to be used
cntry = "Italy"
cntry_act = cntry+"_Actuals"
cntry_pred_m1 = cntry+"_Pred_M1"
cntry_pred_m2 = cntry+"_Pred_M2"


# In[351]:


# 2 Models are created and hence 2 copies of df_train and test to perform the analysis
y_train = df_train[cntry].copy(deep=True)
X_train = df_train[["Day"]].copy(deep=True)

y_test = df_test[cntry].copy(deep=True)
X_test = df_test[["Day"]].copy(deep=True)


# ## Build the LR Model on Training Set

# ### Parameter Tuning and Selection of Degree

# In[352]:


# Populate the results for different degrees
deg = 8
eval_reg(X_train,y_train,deg)


# ### Build the Model using the selected degree

# In[353]:


# Build the model with optimal degree.
degree = 4

reg = Lasso(random_state=rndm_stat)
# create an instance using the optimal degree
lm = make_pipeline(PolynomialFeatures(degree), reg)

# fit the model using training data
lm.fit(X_train, y_train)


# ## Predictions on the train set and evaluate the metrics

# In[354]:


y_train_pred = lm.predict(X_train)


# ## Prediction and Evaluation on the Test Set

# * Make predictions on the test set (y_test_pred)
# * evaluate the model, r-squared on the test set

# ### Predict on Test Data

# In[355]:


# predict y_test_pred based on our model
y_test_pred = lm.predict(X_test)


# In[356]:


y_test_pred


# ### Model Evalution of Metrics of Test Data

# In[357]:


# calculate the RSquared and RMSE for test data and Predicted data
print(color.BOLD+"\nModel Evalutation metrics of test set with degree ",degree)
rsqr,mar,mse,rmse = lm_metrics(y_test,y_test_pred)

print(color.BOLD+"RSquare of the Model is ",round(rsqr,2))
print(color.BOLD+"Mean Absolute Error of the Model is",round(mar,2))
print(color.BOLD+"MSE of the model is ",round(mse,2))
print(color.BOLD+"RMSE of the model is ",round(rmse,2))


# #### Distribution of Actual Test Data and Predicted Test Data

# In[358]:


# Plot the distribution of Actual values of Price and Predicted values of Price
plt.figure(1,figsize=(10,4))
sns.set(style="whitegrid",font_scale=1)
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Values of COVID Cases")
sns.distplot(y_test_pred, hist=False, color="b", label="Predicted Values of COVID Cases" , ax=ax1)
sns.distplot((y_test_pred+rmse), hist=False, color="y", label="Predicated Values of Price + RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
sns.distplot((y_test_pred-rmse), hist=False, color="y", label="Predicated Values of Price - RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
plt.title('LR Model I - Distribution of Actual Values of COVID Cases and Predicted Values of COVID Cases',fontsize=12)
plt.show()


# ### Predict on Actual Test Data

# In[359]:


# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)


# In[360]:


# Create a column with Dates and Day. Starting date is 2020-01-22
covid_df[cntry_pred_m1] = np.rint(y_act_test_pred)


# ### Model 2 with different polynomial

# In[361]:


# Build the model with optimal degree.
degree = 5
reg = Lasso(random_state=rndm_stat)
lm = make_pipeline(PolynomialFeatures(degree), reg)
lm.fit(X_train, y_train)
# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)
# add it to df
covid_df[cntry_pred_m2] = np.rint(y_act_test_pred)


# In[362]:


# add the data to final df
covid_df[cntry] = c_df[cntry]


# In[363]:


covid_df[["Date","WW_Pred_M1",cntry_pred_m1,cntry_pred_m2]].tail(10)


# ## __Plot the Actual vs Predicted Models__

# In[364]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m1],mode='lines+markers',name=cntry_pred_m1))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m2],mode='lines+markers',name=cntry_pred_m2))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=c_df[cntry],mode='lines+markers',name=cntry_act))
fig.update_layout(margin=dict(l=40, r=30, t=25, b=25),) 
fig.update_xaxes(nticks=30)
fig.show()


# # __LR Model using Linear Regression for US__

# ## Explore the data for Modelling

# In[365]:


# Prepare the strings to be used
cntry = "US"
cntry_act = cntry+"_Actuals"
cntry_pred_m1 = cntry+"_Pred_M1"
cntry_pred_m2 = cntry+"_Pred_M2"


# In[366]:


# 2 Models are created and hence 2 copies of df_train and test to perform the analysis
y_train = df_train[cntry].copy(deep=True)
X_train = df_train[["Day"]].copy(deep=True)

y_test = df_test[cntry].copy(deep=True)
X_test = df_test[["Day"]].copy(deep=True)


# ## Build the LR Model on Training Set

# ### Parameter Tuning and Selection of Degree

# In[367]:


# Populate the results for different degrees
deg = 20
eval_reg(X_train,y_train,deg)


# ### Build the Model using the selected degree

# In[368]:


# Build the model with optimal degree.
degree = 14

reg = Lasso(random_state=rndm_stat)
# create an instance using the optimal degree
lm = make_pipeline(PolynomialFeatures(degree), reg)

# fit the model using training data
lm.fit(X_train, y_train)


# ## Predictions on the train set and evaluate the metrics

# In[369]:


y_train_pred = lm.predict(X_train)


# ## Prediction and Evaluation on the Test Set

# * Make predictions on the test set (y_test_pred)
# * evaluate the model, r-squared on the test set

# ### Predict on Test Data

# In[370]:


# predict y_test_pred based on our model
y_test_pred = lm.predict(X_test)


# In[371]:


y_test_pred


# ### Model Evalution of Metrics of Test Data

# In[372]:


# calculate the RSquared and RMSE for test data and Predicted data
print(color.BOLD+"\nModel Evalutation metrics of test set with degree ",degree)
rsqr,mar,mse,rmse = lm_metrics(y_test,y_test_pred)

print(color.BOLD+"RSquare of the Model is ",round(rsqr,2))
print(color.BOLD+"Mean Absolute Error of the Model is",round(mar,2))
print(color.BOLD+"MSE of the model is ",round(mse,2))
print(color.BOLD+"RMSE of the model is ",round(rmse,2))


# #### Distribution of Actual Test Data and Predicted Test Data

# In[373]:


# Plot the distribution of Actual values of Price and Predicted values of Price
plt.figure(1,figsize=(10,4))
sns.set(style="whitegrid",font_scale=1)
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Values of COVID Cases")
sns.distplot(y_test_pred, hist=False, color="b", label="Predicted Values of COVID Cases" , ax=ax1)
sns.distplot((y_test_pred+rmse), hist=False, color="y", label="Predicated Values of Price + RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
sns.distplot((y_test_pred-rmse), hist=False, color="y", label="Predicated Values of Price - RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
plt.title('LR Model I - Distribution of Actual Values of COVID Cases and Predicted Values of COVID Cases',fontsize=12)
plt.show()


# ### Predict on Actual Test Data

# In[374]:


# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)


# In[375]:


# Create a column with Dates and Day. Starting date is 2020-01-22
covid_df[cntry_pred_m1] = np.rint(y_act_test_pred)


# ### Model 2 with different polynomial

# In[376]:


# Build the model with optimal degree.
degree = 13
reg = Lasso(random_state=rndm_stat)
lm = make_pipeline(PolynomialFeatures(degree), reg)
lm.fit(X_train, y_train)
# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)
# add it to df
covid_df[cntry_pred_m2] = np.rint(y_act_test_pred)


# In[377]:


# add the data to final df
covid_df[cntry] = c_df[cntry]


# In[378]:


covid_df[["Date","WW_Pred_M1",cntry_pred_m1,cntry_pred_m2]].tail(10)


# ## __Plot the Actual vs Predicted Models__

# In[379]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m1],mode='lines+markers',name=cntry_pred_m1))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m2],mode='lines+markers',name=cntry_pred_m2))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=c_df[cntry],mode='lines+markers',name=cntry_act))
fig.update_layout(margin=dict(l=40, r=30, t=25, b=25),) 
fig.update_xaxes(nticks=30)
fig.show()


# # __LR Model using Linear Regression for - Spain__

# ## Explore the data for Modelling

# In[380]:


# Prepare the strings to be used
cntry = "Spain"
cntry_act = cntry+"_Actuals"
cntry_pred_m1 = cntry+"_Pred_M1"
cntry_pred_m2 = cntry+"_Pred_M2"


# In[381]:


# 2 Models are created and hence 2 copies of df_train and test to perform the analysis
y_train = df_train[cntry].copy(deep=True)
X_train = df_train[["Day"]].copy(deep=True)

y_test = df_test[cntry].copy(deep=True)
X_test = df_test[["Day"]].copy(deep=True)


# ## Build the LR Model on Training Set

# ### Parameter Tuning and Selection of Degree

# In[382]:


# Populate the results for different degrees
deg = 20
eval_reg(X_train,y_train,deg)


# ### Build the Model using the selected degree

# In[383]:


# Build the model with optimal degree.
degree = 16

reg = Lasso(random_state=rndm_stat)
# create an instance using the optimal degree
lm = make_pipeline(PolynomialFeatures(degree), reg)

# fit the model using training data
lm.fit(X_train, y_train)


# ## Predictions on the train set and evaluate the metrics

# In[384]:


y_train_pred = lm.predict(X_train)


# ## Prediction and Evaluation on the Test Set

# * Make predictions on the test set (y_test_pred)
# * evaluate the model, r-squared on the test set

# ### Predict on Test Data

# In[385]:


# predict y_test_pred based on our model
y_test_pred = lm.predict(X_test)


# In[386]:


y_test_pred


# ### Model Evalution of Metrics of Test Data

# In[387]:


# calculate the RSquared and RMSE for test data and Predicted data
print(color.BOLD+"\nModel Evalutation metrics of test set with degree ",degree)
rsqr,mar,mse,rmse = lm_metrics(y_test,y_test_pred)

print(color.BOLD+"RSquare of the Model is ",round(rsqr,2))
print(color.BOLD+"Mean Absolute Error of the Model is",round(mar,2))
print(color.BOLD+"MSE of the model is ",round(mse,2))
print(color.BOLD+"RMSE of the model is ",round(rmse,2))


# #### Distribution of Actual Test Data and Predicted Test Data

# In[388]:


# Plot the distribution of Actual values of Price and Predicted values of Price
plt.figure(1,figsize=(10,4))
sns.set(style="whitegrid",font_scale=1)
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Values of COVID Cases")
sns.distplot(y_test_pred, hist=False, color="b", label="Predicted Values of COVID Cases" , ax=ax1)
sns.distplot((y_test_pred+rmse), hist=False, color="y", label="Predicated Values of Price + RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
sns.distplot((y_test_pred-rmse), hist=False, color="y", label="Predicated Values of Price - RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
plt.title('LR Model I - Distribution of Actual Values of COVID Cases and Predicted Values of COVID Cases',fontsize=12)
plt.show()


# ### Predict on Actual Test Data

# In[389]:


# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)


# In[390]:


# Create a column with Dates and Day. Starting date is 2020-01-22
covid_df[cntry_pred_m1] = np.rint(y_act_test_pred)


# ### Model 2 with different polynomial

# In[391]:


# Build the model with optimal degree.
degree = 15
reg = Lasso(random_state=rndm_stat)
lm = make_pipeline(PolynomialFeatures(degree), reg)
lm.fit(X_train, y_train)
# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)
# add it to df
covid_df[cntry_pred_m2] = np.rint(y_act_test_pred)


# In[392]:


# add the data to final df
covid_df[cntry] = c_df[cntry]


# In[393]:


covid_df[["Date","WW_Pred_M1",cntry_pred_m1,cntry_pred_m2]].tail(10)


# ## __Plot the Actual vs Predicted Models__

# In[394]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m1],mode='lines+markers',name=cntry_pred_m1))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m2],mode='lines+markers',name=cntry_pred_m2))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=c_df[cntry],mode='lines+markers',name=cntry_act))
fig.update_layout(margin=dict(l=40, r=30, t=25, b=25),) 
fig.update_xaxes(nticks=30)
fig.show()


# # __LR Model using Linear Regression for Germany__

# ## Explore the data for Modelling

# In[395]:


# Prepare the strings to be used
cntry = "Germany"
cntry_act = cntry+"_Actuals"
cntry_pred_m1 = cntry+"_Pred_M1"
cntry_pred_m2 = cntry+"_Pred_M2"


# In[396]:


# 2 Models are created and hence 2 copies of df_train and test to perform the analysis
y_train = df_train[cntry].copy(deep=True)
X_train = df_train[["Day"]].copy(deep=True)

y_test = df_test[cntry].copy(deep=True)
X_test = df_test[["Day"]].copy(deep=True)


# ## Build the LR Model on Training Set

# ### Parameter Tuning and Selection of Degree

# In[397]:


# Populate the results for different degrees
deg = 9
eval_reg(X_train,y_train,deg)


# ### Build the Model using the selected degree

# In[398]:


# Build the model with optimal degree.
degree = 7

reg = Lasso(random_state=rndm_stat)
# create an instance using the optimal degree
lm = make_pipeline(PolynomialFeatures(degree), reg)

# fit the model using training data
lm.fit(X_train, y_train)


# ## Predictions on the train set and evaluate the metrics

# In[399]:


y_train_pred = lm.predict(X_train)


# ## Prediction and Evaluation on the Test Set

# * Make predictions on the test set (y_test_pred)
# * evaluate the model, r-squared on the test set

# ### Predict on Test Data

# In[400]:


# predict y_test_pred based on our model
y_test_pred = lm.predict(X_test)


# In[401]:


y_test_pred


# ### Model Evalution of Metrics of Test Data

# In[402]:


# calculate the RSquared and RMSE for test data and Predicted data
print(color.BOLD+"\nModel Evalutation metrics of test set with degree ",degree)
rsqr,mar,mse,rmse = lm_metrics(y_test,y_test_pred)

print(color.BOLD+"RSquare of the Model is ",round(rsqr,2))
print(color.BOLD+"Mean Absolute Error of the Model is",round(mar,2))
print(color.BOLD+"MSE of the model is ",round(mse,2))
print(color.BOLD+"RMSE of the model is ",round(rmse,2))


# #### Distribution of Actual Test Data and Predicted Test Data

# In[403]:


# Plot the distribution of Actual values of Price and Predicted values of Price
plt.figure(1,figsize=(10,4))
sns.set(style="whitegrid",font_scale=1)
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Values of COVID Cases")
sns.distplot(y_test_pred, hist=False, color="b", label="Predicted Values of COVID Cases" , ax=ax1)
sns.distplot((y_test_pred+rmse), hist=False, color="y", label="Predicated Values of Price + RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
sns.distplot((y_test_pred-rmse), hist=False, color="y", label="Predicated Values of Price - RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
plt.title('LR Model I - Distribution of Actual Values of COVID Cases and Predicted Values of COVID Cases',fontsize=12)
plt.show()


# ### Predict on Actual Test Data

# In[404]:


# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)


# In[405]:


# Create a column with Dates and Day. Starting date is 2020-01-22
covid_df[cntry_pred_m1] = np.rint(y_act_test_pred)


# ### Model 2 with different polynomial

# In[406]:


# Build the model with optimal degree.
degree = 6
reg = Lasso(random_state=rndm_stat)
lm = make_pipeline(PolynomialFeatures(degree), reg)
lm.fit(X_train, y_train)
# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)
# add it to df
covid_df[cntry_pred_m2] = np.rint(y_act_test_pred)


# In[407]:


# add the data to final df
covid_df[cntry] = c_df[cntry]


# In[408]:


covid_df[["Date","WW_Pred_M1",cntry_pred_m1,cntry_pred_m2]].tail(10)


# ## __Plot the Actual vs Predicted Models__

# In[409]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m1],mode='lines+markers',name=cntry_pred_m1))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m2],mode='lines+markers',name=cntry_pred_m2))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=c_df[cntry],mode='lines+markers',name=cntry_act))
fig.update_layout(margin=dict(l=40, r=30, t=25, b=25),) 
fig.update_xaxes(nticks=30)
fig.show()


# # __LR Model using Linear Regression for India__

# ## Explore the data for Modelling

# In[410]:


# Prepare the strings to be used
cntry = "India"
cntry_act = cntry+"_Actuals"
cntry_pred_m1 = cntry+"_Pred_M1"
cntry_pred_m2 = cntry+"_Pred_M2"


# In[411]:


# 2 Models are created and hence 2 copies of df_train and test to perform the analysis
y_train = df_train[cntry].copy(deep=True)
X_train = df_train[["Day"]].copy(deep=True)

y_test = df_test[cntry].copy(deep=True)
X_test = df_test[["Day"]].copy(deep=True)


# ## Build the LR Model on Training Set

# ### Parameter Tuning and Selection of Degree

# In[412]:


# Populate the results for different degrees
deg = 25
eval_reg(X_train,y_train,deg)


# ### Build the Model using the selected degree

# In[413]:


# Build the model with optimal degree.
degree = 20

reg = Lasso(random_state=rndm_stat)
# create an instance using the optimal degree
lm = make_pipeline(PolynomialFeatures(degree), reg)

# fit the model using training data
lm.fit(X_train, y_train)


# ## Predictions on the train set and evaluate the metrics

# In[414]:


y_train_pred = lm.predict(X_train)


# ## Prediction and Evaluation on the Test Set

# * Make predictions on the test set (y_test_pred)
# * evaluate the model, r-squared on the test set

# ### Predict on Test Data

# In[415]:


# predict y_test_pred based on our model
y_test_pred = lm.predict(X_test)


# In[416]:


y_test_pred


# ### Model Evalution of Metrics of Test Data

# In[417]:


# calculate the RSquared and RMSE for test data and Predicted data
print(color.BOLD+"\nModel Evalutation metrics of test set with degree ",degree)
rsqr,mar,mse,rmse = lm_metrics(y_test,y_test_pred)

print(color.BOLD+"RSquare of the Model is ",round(rsqr,2))
print(color.BOLD+"Mean Absolute Error of the Model is",round(mar,2))
print(color.BOLD+"MSE of the model is ",round(mse,2))
print(color.BOLD+"RMSE of the model is ",round(rmse,2))


# #### Distribution of Actual Test Data and Predicted Test Data

# In[418]:


# Plot the distribution of Actual values of Price and Predicted values of Price
plt.figure(1,figsize=(10,4))
sns.set(style="whitegrid",font_scale=1)
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Values of COVID Cases")
sns.distplot(y_test_pred, hist=False, color="b", label="Predicted Values of COVID Cases" , ax=ax1)
sns.distplot((y_test_pred+rmse), hist=False, color="y", label="Predicated Values of Price + RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
sns.distplot((y_test_pred-rmse), hist=False, color="y", label="Predicated Values of Price - RMSE" , ax=ax1, kde_kws={'linestyle':'--'})
plt.title('LR Model I - Distribution of Actual Values of COVID Cases and Predicted Values of COVID Cases',fontsize=12)
plt.show()


# ### Predict on Actual Test Data

# In[419]:


# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)


# In[420]:


# Create a column with Dates and Day. Starting date is 2020-01-22
covid_df[cntry_pred_m1] = np.rint(y_act_test_pred)


# ### Model 2 with different polynomial

# In[421]:


# Build the model with optimal degree.
degree = 19
reg = Lasso(random_state=rndm_stat)
lm = make_pipeline(PolynomialFeatures(degree), reg)
lm.fit(X_train, y_train)
# predict y_test_pred based on our model
y_act_test_pred = lm.predict(X_act_test)
# add it to df
covid_df[cntry_pred_m2] = np.rint(y_act_test_pred)


# In[422]:


# add the data to final df
covid_df[cntry] = c_df[cntry]


# In[423]:


covid_df[["Date","WW_Pred_M1",cntry_pred_m1,cntry_pred_m2]].tail(10)


# ## __Plot the Actual vs Predicted Models__

# In[424]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m1],mode='lines+markers',name=cntry_pred_m1))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=covid_df[cntry_pred_m2],mode='lines+markers',name=cntry_pred_m2))
fig.add_trace(go.Scatter(x=covid_df["Date"],y=c_df[cntry],mode='lines+markers',name=cntry_act))
fig.update_layout(margin=dict(l=40, r=30, t=25, b=25),) 
fig.update_xaxes(nticks=30)
fig.show()


# In[425]:


covid_df.tail(20)


# In[439]:


from datetime import date
today = date.today()
fp = r"C:\My Folders\OneDrive\Surface\Sadguru\Lakshmi\Study\IIIB_PGDS\Hackathon\COVID_19" + '/' + today.strftime("%b_%d_%Y") + ".csv"
fp


# In[440]:


covid_df.to_csv(fp,index=False)


# In[ ]:




