#!/usr/bin/env python
# coding: utf-8

# # 🚜Predicting the sale price of bulldozer using machine learning 
# 
# In this notebook, we're going to go through an example machine learning project with a goal of predicting the sale price of bulldozers.
# 
# ## 1.Problem Definition
# > How well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similar bulldozers have been sold for?
# 
# ## 2.Data
# The data is downloaded from kaggle Bluebook for Bulldozers competition: https://www.kaggle.com/code/premgaikwad07/time-series-bulldozer-price-prediction/notebook
# There are 3 main datasets:
# * Train.csv is the training set, which contains data through the end of 2011.
# * Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
# * Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.
# 
# ## 3.Evaluation
# The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.
# 
# For more on the evaluation project check: https://www.kaggle.com/competitions/bluebook-for-bulldozers/overview
# 
# **Note:** The goal for most regression evaluation metrics is to minimize the error. For example, our goal for this project will be to build a machine learning model which minimises RMSLE.
# 
# ## 4.Features
# 
# Kaggle provides a data dictionary detailing all of the features of the dataset you can view this data on google sheet:
# https://docs.google.com/spreadsheets/d/1TSGbtf_s5YQyJHzj_1s8zusR01rd7tUrFPiqj2WGKQQ/edit?usp=sharing
# 

# In[136]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[137]:


# Import training and validation sets
df = pd.read_csv("data/bluebook-for-bulldozers/TrainAndValid.csv",low_memory=False)


# In[138]:


df.info


# In[139]:


df.isna().sum()


# In[140]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])


# In[141]:


df.SalePrice.plot.hist()


# ### Parsing dates
# 
# When we work with time series data, we want to enrich the time and date components as much as possible.
# 
# We can do that by telling pandas which of our columns has date in it using a `parse_date` parameter
# 

# In[142]:


#Import date again but this time parse dates

df = pd.read_csv("data/bluebook-for-bulldozers/TrainAndValid.csv",
                 low_memory=False,
                 parse_dates=["saledate"])


# In[143]:


df.saledate.dtype


# In[144]:


df.saledate[:1000]


# In[145]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])


# In[146]:


df.head()


# In[147]:


df.head().T


# In[148]:


df.saledate.head(20)


# ### Sort dataframe by saledate
#  
# when working with time series data, it's a good idea to sort it by date.

# In[149]:


# Sort dataframe in date order
df.sort_values(by=["saledate"],inplace=True, ascending=True)
df.saledate.head(20)


# ### Make copy of the original dataframe
# 
# We make the copy of original dataframe so when we manipulate the copy, we still have that original data
# 

# In[150]:


# Make a copy

df_tmp = df.copy()


# In[151]:


df_tmp.saledate.head(20)


# In[152]:


df_tmp.head().T


# ### Add datetime  parameters for `saledate` column

# In[153]:


df_tmp["SaleYear"] = df_tmp.saledate.dt.year
df_tmp["SaleMonth"] = df_tmp.saledate.dt.month
df_tmp["SaleDay"] = df_tmp.saledate.dt.day
df_tmp["SaleDayOfWeek"] = df_tmp.saledate.dt.dayofweek
df_tmp["SaleOfYear"] = df_tmp.saledate.dt.dayofyear


# In[154]:


# now we've enriched our dataframe with date time feature we can remove saledate

df_tmp.drop("saledate",axis=1 , inplace= True)


# In[155]:


# doing EDA 

df_tmp.state.value_counts()


# ## 5. Modelling
# 
# We've done enough EDA (we could always do more) but let's start to do some mdoel-driven EDA.

# ### Converting String into categories
# 
# one way we can turn all of our data into numbers is by converting them into pandas categories
# 
# we can check the different datatypes compatible with pandas here:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Categorical.html

# In[156]:


pd.api.types.is_object_dtype(df_tmp["UsageBand"])


# In[157]:


# Find the columns which contain strings

for label, content in df_tmp.items():
    if pd.api.types.is_object_dtype(content):
        print(label)


# In[158]:


# if you wondering what does df.items() does, here is an example

random_dict={
    "key1": "Hello",
    "key2": "World!"
}
for key, value in random_dict.items():
    print(f"this is a key: {key}",
          f"this is a value: {value}")


# In[159]:


# This will turn all the string value into category value

for label, content in df_tmp.items():
    if pd.api.types.is_object_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()


# In[160]:


df_tmp.info()


# In[161]:


df_tmp.state.cat.categories


# In[162]:


df_tmp.state.value_counts()


# In[163]:


df_tmp.state.cat.codes


# Thanks to pandas Categories we now have a way to access all of our data in the form of numbers.
# 
# But we still have a bunch of missing data... 

# In[164]:


# Check missing data
df_tmp.isnull().sum()/len(df_tmp)


# ### Save preprocessed data

# In[165]:


# Export current tmp dataframe

df_tmp.to_csv("data/bluebook-for-bulldozers/train_tmp.csv" , index = False)


# In[166]:


# Import preprocessed data
df_tmp = pd.read_csv("data/bluebook-for-bulldozers/train_tmp.csv",
              low_memory = False)
df_tmp.head().T


# In[167]:


df_tmp.isna().sum()


# ## Filling Missing values
# 
# ### Fill numeric missing values first

# In[168]:


for label,content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)


# In[169]:


df_tmp.ModelID


# In[170]:


# Check for which numeric column have null values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[171]:


# Fill numeric rows with the median
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells us if the data was missing or not
            df_tmp[label+ "_is_missing"] = pd.isnull(content)
            # Fill missing value with median 
            df_tmp[label] = content.fillna(content.median())


# In[172]:


# Demonstrate the how median is how more robust than mean

hundreds = np.full((1000,),100)
hundreds_billion = np.append(hundreds, 1000000000)
np.mean(hundreds), np.mean(hundreds_billion), np.median(hundreds), np.median(hundreds_billion)


# In[173]:


# Check if there's any null numeric values
for label,content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[174]:


# Check to see how many examples are missing

df_tmp.auctioneerID_is_missing.value_counts()


# In[175]:


df_tmp.isna().sum()


# ### Filling and turning categorical variable into numbers

# In[176]:


# Check for columns which are not numeric
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[177]:


# Turn categorical variables into numbers and fill missing
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing values 
        df_tmp[label+"_is_missing"]=pd.isnull(content)
        # Turn categories into numbers and add +1 
        df_tmp[label] = pd.Categorical(content).codes + 1


# In[178]:


pd.Categorical(df_tmp["state"]).codes


# In[179]:


pd.Categorical(df_tmp["UsageBand"]).codes


# In[180]:


# There are -1 that we doesn't want and pandas do that -1 thing so we add +1 so that it remains same

pd.Categorical(df_tmp["UsageBand"]).codes+1


# In[181]:


df_tmp.info()


# In[182]:


df_tmp.head().T


# In[183]:


df_tmp.isna().sum()


# Now that all of data is numeric as well as our dataframe has no missing values, we should be able to build a machine learning model.

# In[184]:


len(df_tmp)


# In[185]:


get_ipython().run_cell_magic('time', '', '# Instantiate model \nfrom sklearn.ensemble import RandomForestRegressor\n\nmodel=RandomForestRegressor (n_jobs=-1,\n                            random_state=42)\n\n# Fit the Model\nmodel.fit(df_tmp.drop("SalePrice",axis=1),df_tmp["SalePrice"]) \n')


# In[186]:


# Score the Model
model.score(df_tmp.drop("SalePrice",axis=1),df_tmp["SalePrice"])


# **Question:** Why doesn't the above matric hold water? (why isn't the metric reliable )

# ### Splitting data into train/validation sets

# In[187]:


df_tmp.head()


# In[188]:


df_tmp.SaleYear.value_counts()


# In[189]:


# Split data into training and validation
df_val = df_tmp[df_tmp.SaleYear == 2012]
df_train = df_tmp[df_tmp.SaleYear != 2012]

len(df_val), len(df_train)


# In[190]:


# Split data into X and Y
X_train, y_train = df_train.drop("SalePrice",axis=1),df_train.SalePrice
X_valid, y_valid = df_val.drop("SalePrice",axis=1),df_val.SalePrice

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# In[191]:


y_train


# ### Build a Evaluation Function

# In[192]:


# Create Evaluation Function (The Competition uses RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rsmle(y_test, y_preds):
    '''
    Calculates Root Mean Squared Log Error between predictions and true labels.
    '''
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate model on few different levels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds =  model.predict(X_valid)
    scores={"Training MAE": mean_absolute_error(y_train,train_preds),
            "Valid MAE": mean_absolute_error(y_valid,val_preds),
            "Training RMSLE": rsmle(y_train,train_preds),
            "Valid RMSLE": rsmle(y_valid,val_preds),
            "Training R^2": r2_score(y_train,train_preds),
            "Valid R^2": r2_score(y_valid,val_preds)}
    return scores


# ## Testing our model on a subset(to tune the hyperparameters)

# In[193]:


# #This takes far too long... for experimenting 

# %%time

# model= RandomForestRegressor(n_jobs=-1,
#                              random_state=42)

# model.fit(X_train, y_train)


# In[194]:


len(X_train)


# In[195]:


# Change max_Sample value
model= RandomForestRegressor(n_jobs=-1,
                            random_state=42,
                            max_samples=10000)


# In[196]:


get_ipython().run_cell_magic('time', '', '# Cutting down on the max number of samples each estimator can see improves training time \nmodel.fit(X_train,y_train)\n')


# In[197]:


X_train.shape[0] * 100 / 1000000


# In[198]:


100000 * 100


# In[199]:


show_scores(model)


# # Hyperparameter tuning with RandomizedSearchCV

# In[200]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import RandomizedSearchCV\n\n# Different RandomForestRegressor hyperparameters\nrf_grid = {"n_estimators":np.arange(10,100,10),\n           "max_depth": [None,3,5,10],\n           "min_samples_split": np.arange(2,20,2),\n           "min_samples_leaf": np.arange(1,20,2),\n           "max_features": [0.5,1,"sqrt","auto"],\n           "max_samples":[10000]}\n\n# Instantiate RandomizedSearchCV model\nrs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,\n                                                   random_state=42),\n                             param_distributions=rf_grid,\n                             n_iter=2,\n                             cv=5,\n                             verbose=True)\n\n# Fir the RandomizedSearchCV model\nrs_model.fit(X_train,y_train)\n')


# In[201]:


# Find the best model hyperparameters
rs_model.best_params_


# In[202]:


# Evaluate the RandomizedSearcdCV model
show_scores(rs_model)


# ### Train a model with the best hyperparameters
# 
# **Note:** These were found after 100 iterations of `RandomizedSearchCV`.

# In[203]:


get_ipython().run_cell_magic('time', '', '\n# Most Ideal HyperParameters\nideal_model = RandomForestRegressor(n_estimators=40,\n                                   min_samples_leaf=1,\n                                   min_samples_split=14,\n                                   max_features=0.5,\n                                   n_jobs=-1,\n                                   max_samples=None,\n                                   random_state=42) # random state so our results are reproducible\n\n# Fit the ideal model\nideal_model.fit(X_train,y_train)\n')


# In[204]:


# Scores on ideal_model (trained on all the data)
show_scores(ideal_model)


# In[205]:


# Scores on rs_model (only trained on ~10,000 examples)
show_scores(rs_model)


# ### Making Predictions on Test data

# In[241]:


# Import the test data
df_test = pd.read_csv("data/bluebook-for-bulldozers/Test.csv",
                       low_memory=False,
                       parse_dates=["saledate"])
df_test.head()


# In[242]:


df_test.isna().sum()


# In[243]:


df_test.info()


# In[244]:


df_test.columns


# In[245]:


X_train.columns


# ### Preprocessing the data (getting the test dataset in the same format as our training dataset)

# In[246]:


def preprocess_data(df):
    """
    Performs transformation on df and returns transformed df.
    """
    df["SaleYear"] = df.saledate.dt.year
    df["SaleMonth"] = df.saledate.dt.month
    df["SaleDay"] = df.saledate.dt.day
    df["SaleDayOfWeek"] = df.saledate.dt.dayofweek
    df["SaleOfYear"] = df.saledate.dt.dayofyear
    
    df.drop("saledate",axis=1, inplace=True)
    
    # Fill the numeric values with the median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if...
                df[label+'_is_missing'] = pd.isnull(content)
                # Fill missing numeric values with mdeian
                df[label] = content.fillna(content.median())
            
    # Filed categorical missing data and turn categories...
        if not pd.api.types.is_numeric_dtype(content):
            df[label+'_is_missing'] = pd.isnull(content)
            # We add +1 to the category code because ...
            df[label] = pd.Categorical(content).codes + 1 
    return df


# In[247]:


# Process the test data 
df_test = preprocess_data(df_test)
df_test.head()


# In[248]:


# We can find how columns differ using a python sets
set(X_train.columns) - set(df_test.columns)


# In[249]:


# Manually adjust df_test to have auctioneerID_is_missing column
df_test["auctioneerID_is_missing"] = False
df_test.head()


# Finally now our test dataframe has the same features as our training dataframe, we can make pridictions!

# In[254]:


# Make Predictions on updated test data
df_test = df_test[X_train.columns]
test_preds = ideal_model.predict(df_test)


# In[253]:


test_preds


# We've made some preditions but they're not in the same format kaggle is asking for:
# https://www.kaggle.com/competitions/bluebook-for-bulldozers/overview/evaluation
#     

# In[220]:


# Format Predicions into the same format Kaggle is after
df_preds =pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalePrice"] = test_preds
df_preds


# In[229]:


# Export Prediction data
df_preds.to_csv("data/bluebook-for-bulldozers/test_predictions.csv",index=False)


# ### Feature Importance 
# Feature Importance seeks to figure out which different attributes of the data were most importance when it comes to predicting the **target variable** (SalePrice).

# In[263]:


# Find feature importance of our best model 
ideal_model.feature_importances_


# In[275]:


def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
          .sort_values("feature_importances", ascending=False)
          .reset_index(drop=True))
    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:n])  # corrected the column name here
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature Importance")
    ax.invert_yaxis()


# In[276]:


plot_features(X_train.columns, ideal_model.feature_importances_)


# In[281]:


df["Enclosure"].value_counts()


#  **Question to finish:** Why might knowing the feature importances of the trained machine learning model be helpful?
#  
#  **Final Challenge:** What other machine learning models could you try on our dataset?
#  **HINT:** https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
#  Check out the regression section of this map, or try to look at something like CatBoost.ai or XGBoost.ai 

# In[ ]:




