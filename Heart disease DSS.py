#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
# from collections import Counter

import xlsxwriter

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import matplotlib.pyplot as plt
# %matplotlib inline


# In[2]:


# def getType():
#     global max_days
#     if MaxDays.get() == "":
#         max_days = "3,10"
#     else:
#         max_days = MaxDays.get()

def Heart_Disease_Data():
    global dfSM
    print('Decision Support System(DSS) :-)')
    import_file_path = filedialog.askopenfilename()
    print('Importing...')
    print('Please Wait :)')
    missing_values = ["n/a", "na", "--", "nan", ".", "-", "- ", "_"]
    dfSM = pd.read_excel(import_file_path, na_values=missing_values, engine='openpyxl')
    print('Decision Support System(DSS) has been uploaded')

def To_predictData():
    global preData
    print('Data to predict :-)')
    import_file_path = filedialog.askopenfilename()
    print('Importing...')
    print('Please Wait :)')
    missing_values = ["n/a", "na", "--", "nan", ".", "-", "- ", "_"]
    preData = pd.read_excel(import_file_path, na_values=missing_values, engine='openpyxl')
    print('Data to predict has been uploaded')
    print('Files have been Imported\nClick Run and close the window!')


# In[3]:


print('Please Import Excel Files')

root = tk.Tk()
root.geometry('550x400')
root.title("Decision Support System(DSS)")

label_0 = tk.Label(root, text="Heart Disease Recognition", relief="solid", width=30, font=("arial", 19, "bold"))
label_0.place(x=300, y=100, anchor="center")

# MaxDays = tk.StringVar()
# LMaxDays = tk.Label(root, text="Max Days: ", width=10, font=("bold", 10))
# LMaxDays.place(x=175, y=270, anchor="center")
# EMaxDays = tk.Entry(root, textvar=MaxDays, width=5)
# EMaxDays.place(x=235, y=270, anchor="center")

# max_days = None
dfSM = None
preData = None
# df_scheme= None
# df_ride= None

Get_List = tk.Button(root, text='Heart Disease Data', width=20, bg='brown', fg='white',
                     command=Heart_Disease_Data).place(x=300, y=200, anchor="center")

Get_List = tk.Button(root, text='Data to predict', width=20, bg='brown', fg='white',
                     command=To_predictData).place(x=300, y=250, anchor="center")

Get_Max = tk.Button(root, text='Run', width=15, bg='green', fg='white').place(x=300, y=350, anchor="center")

root.mainloop()

# del Get_List
# del Get_Ride
# del Get_Max
# del Get_DailyFraud
# del Get_HistFraud


# In[4]:


y = dfSM.iloc[:,-1].values
X = dfSM.drop('Target',axis=1).values

pso = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
X_selected_features = X[:,pso==1]


# In[5]:


XSM = dfSM.drop('Target', axis=1)
XSM = XSM.drop(XSM.columns[list(pso).index(0)],axis=1)
ySM = dfSM['Target']

SS = StandardScaler()
SS.fit(X_selected_features)
X_normal_SM = SS.transform(X_selected_features)
# X_normal_SM = X_selected_features.copy()

X_normal_SM = pd.DataFrame(X_normal_SM)
dfSM = pd.concat([X_normal_SM, ySM], axis=1)


# In[6]:


col = list(set(XSM.columns))
col.append('Target')
dfSM.columns = col


# In[7]:


dfSM.shape


# In[8]:


CLF = XGBClassifier()
# CLF = RandomForestClassifier()
CLF.fit(X_normal_SM, y)


# In[9]:


try:
    preData = preData.drop('Unnamed: 0', axis=1)
except:
    1+1


# In[10]:


preData = preData.rename(columns={
                       'جنس': 'Sex',
                       'سن': 'Age',
                       'C':'Target',
                       'تعداد روزهای بستری':'Number of hospitalization days',
                       'نبض':'Heart rate',
                       'دیابت':'Diabetes',
                       'فشار خون سیستولیک':'BPs',
                       'فشار خون دیاستولیک':'BPd',
                      'درد قفسه سینه':'Chest pain',
                      'سیگار':'Smoke',
                      'سابقه تنگی نفس':'History of shortness of breath',
                      'سابقه بیماری قلبی':'History of heart disease',
                      'سابقه فامیلی بیماری قلبی':'Family history of heart disease',
                      'سابقه فشار خون':'History of high blood pressure'})


# In[11]:


preData['Sex'][preData['Sex']=='زن']=0
preData['Sex'][preData['Sex']=='مرد']=1
preData['Sex'][preData['Sex']=='مرد ']=1
preData['Sex'] = preData['Sex'].astype('int64')


# In[12]:


def  approach2_impute_metric(messy_df, baseData, metric):
    # Finding columns which have null values
    colnames = []
    for col in messy_df.columns:
        if messy_df[col].isnull().sum() > 0:
            colnames.append(col)
    if len(colnames) == 0:
        return messy_df, []
    # Create X_df of predictor columns
    X_df = messy_df.drop(colnames, axis = 1)
    
    # Create Y_df of predicted columns
    Y_df = messy_df[colnames]
        
    # Create empty dataframes and list
    Y_pred_df = pd.DataFrame(columns=colnames)
    Y_missing_df = pd.DataFrame(columns=colnames)
    missing_list = []
    
    # Loop through all columns containing missing values
    for col in messy_df[colnames]:
        
        # Number of missing values in the column
        missing_count = messy_df[col].isnull().sum()
        
#         # Separate train dataset which does not contain missing values
#         messy_df_train = messy_df[~messy_df[col].isnull()]
        
        # Create X and Y within train dataset
        msg_cols_train_df = baseData[col]
        messy_df_train = baseData.drop(colnames, axis=1)

        # Create test dataset, containing missing values in Y    
        messy_df_test = messy_df[messy_df[col].isnull()]
        
        # Separate X and Y in test dataset
        msg_cols_test_df = messy_df_test[col]
        messy_df_test = messy_df_test.drop(colnames,axis = 1)

        # Copy X_train and Y_train
        Y_train = msg_cols_train_df.copy()
        X_train = messy_df_train.copy()
        
        # Linear Regression model
        if metric == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, Y_train)
            print("R-squared value is: " + str(model.score(X_train, Y_train)))
          
        # Random Forests regression model
        elif metric == "Random Forests":
            model = RandomForestRegressor(n_estimators = 50 , oob_score = True, max_depth=10)
            model.fit(X_train, Y_train) 
                   
        X_test = messy_df_test.copy()
        # Predict Y_test values by passing X_test as input to the model
        Y_test = model.predict(X_test)
        
        Y_test_integer = pd.to_numeric(pd.Series(Y_test),downcast='integer')
        
        # Append predicted Y values to known Y values
        Y_complete = Y_train.append(Y_test_integer)
        Y_complete = Y_complete.reset_index(drop = True)
        
        # Update list of missing values
        missing_list.append(Y_test.tolist())
        
        Y_pred_df[col] = Y_complete
        Y_pred_df = Y_pred_df.reset_index(drop = True)
    
    # Create cleaned up dataframe
    clean_df = X_df.join(Y_pred_df)
    
    return clean_df,missing_list


# In[13]:


dfSM.head()


# In[14]:


try:
    dfSM = dfSM.drop('Target', axis=1) 
except:
    1+1
preData = preData[dfSM.columns]
preData .head()


# In[15]:


dfSM


# In[16]:


cleanDF, mislist = approach2_impute_metric(preData, dfSM , "Random Forests")
cleanDF.shape


# In[17]:


StandardizedData = SS.transform(cleanDF)
y = CLF.predict(StandardizedData)

# y = CLF.predict(cleanDF.values[:,pso==1])
y


# In[18]:


newDF = cleanDF.copy()
newDF['predict'] = y
newDF.to_excel('Result.xlsx', index=False)


# In[19]:


# cleanDF = cleanDF.drop('predict', axis=1)

