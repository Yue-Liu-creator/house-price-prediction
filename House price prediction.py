#!/usr/bin/env python
# coding: utf-8

# 
# # this project is for prediction of the house price

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# totally there are 1460 rows and 81 columns


# In[3]:


data=pd.read_csv("train.csv")


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data.info()


# In[7]:


data.isnull().sum().sort_values()


# ### dealling with missing values

# In[8]:


# drop  the columns that have the missing value more than 5%, baesd on that we can drop 11 columns
drop_columns1=data.isnull().sum()[data.isnull().sum()>data.shape[0]*0.05].index
drop_columns1


# In[9]:


data.drop(drop_columns1,axis=1,inplace= True)


# In[10]:


# as ID is the column that used by kaggle, we can drop this column
data.drop("Id",axis=1,inplace=True)


# In[11]:


# as for other missing value, we will imputate the categorical data with mode and numeric data with mean


# In[12]:


missing_data_index=data.isnull().sum()[data.isnull().sum().sort_values()>0].index
missing_data_index


# In[13]:


numeric= data.select_dtypes(include=["int64","float64"])
numeric=numeric.fillna(numeric.mean())
categorical= data.select_dtypes(include=["object"])
missing_index = categorical.isnull().sum()[categorical.isnull().sum()>0].index
for index in missing_index:
    
    categorical[index]=categorical[index].fillna(categorical[index].value_counts().index[0])


# In[14]:


data.isnull().sum()


# In[15]:


# recombine the data


# In[16]:


data=pd.concat([categorical,numeric],axis=1)


# ### create the new features 

# In[17]:


data["since_RemodAdd"]=data["YrSold"]-data["YearRemodAdd"]


# In[18]:


data[data["since_RemodAdd"]<0]


# In[19]:


data=data.drop(index= 523,axis=0)


# In[20]:


data["since_Built"]=data["YrSold"]-data['YearBuilt']


# In[21]:


data[data["since_RemodAdd"]<0]


# In[22]:


# we can drop the original date for the built year and remod year


# In[23]:


data=data.drop(["YearBuilt","YearRemodAdd"],axis=1)


# In[24]:


#drop all the columns that ;eak the information about selling


# In[25]:


data=data.drop(["YrSold","MoSold","SaleType","SaleCondition"],axis=1)


# ### transforming the target variable

# In[26]:


sns.distplot(data["SalePrice"])


# In[27]:


sns.boxplot(data["SalePrice"])


# In[28]:


# this is a typical right-skewed distribution


# In[29]:


data["log_SalePrice"]= np.log(data["SalePrice"])


# In[30]:


sns.distplot(data["log_SalePrice"])


# In[31]:


sns.boxplot(data["log_SalePrice"])


# ### data cleaning

# In[32]:


# transforming ordinal variables 15 
ordinal_list=["LotShape","Utilities","LandContour","PavedDrive","Functional","KitchenQual","Electrical","HeatingQC","BsmtFinType2","BsmtFinType1","BsmtExposure","BsmtCond","BsmtQual", "ExterCond", "ExterQual"]

len(ordinal_list)


# In[33]:


from  sklearn.preprocessing import LabelEncoder


# In[34]:


transformed_ordinal={}
for ordinal in ordinal_list:
    label=data[ordinal].value_counts().index
    encoder= LabelEncoder()
    encoder.fit(label)
    result=encoder.transform(data[ordinal])
    transformed_ordinal[ordinal]=result


# In[35]:


transformed_ordinal


# In[36]:


transformed_frame=pd.DataFrame(transformed_ordinal)


# In[37]:


data=data.drop(ordinal_list,axis=1)


# In[38]:


data=pd.concat([data,transformed_frame],axis=1)


# In[39]:


object_columns=data.select_dtypes(include=["object"]).columns


# In[40]:


for i in object_columns:
    print(data[i].value_counts() ) 


# In[41]:


data.shape[0]*0.70


# In[42]:


drop_list=[]
for i in object_columns:
    if data[i].value_counts()[0]>data.shape[0]*0.70:
        drop_list.append(i)

drop_list


# In[43]:


data=data.drop(drop_list,axis=1)


# In[44]:


data.select_dtypes(include=["object"]).columns


# In[45]:


fig = sns.FacetGrid(data,hue='Neighborhood',aspect=4)
fig.map(sns.kdeplot,'log_SalePrice',shade=True)

upper = data['log_SalePrice'].max()
lower = data['log_SalePrice'].min()
fig.set(xlim=(lower,upper))

fig.add_legend()


# In[46]:


# in order to avoid the influence from the extreme point, we use median instead of mean to represent each variable
neighbor_class=data.groupby(by="Neighborhood").agg([np.median])["log_SalePrice"].sort_values("median")
neighbor_class


# In[47]:


Price_info=data.describe()["log_SalePrice"]
Price_info


# In[48]:


class1=neighbor_class[neighbor_class["median"]<Price_info["25%"]].index
class2 = neighbor_class[(Price_info["25%"]<=neighbor_class["median"])&(neighbor_class["median"] <Price_info["50%"])].index
class3= neighbor_class[(Price_info["50%"]<=neighbor_class["median"])& (neighbor_class["median"]<Price_info["75%"])].index
class4 = neighbor_class[(Price_info["75%"]<=neighbor_class["median"])].index


# In[49]:


def assign_class(row):
    if row in class1:
        row=1
    elif row in class2:
        row=2
    elif row in class3:
        row=3
    else:
        row=4
    return row


# In[50]:


data.Neighborhood=data.Neighborhood.apply(assign_class)


# In[51]:


data[["Exterior1st","Exterior2nd"]]


# In[52]:


data[data["Exterior1st"]==data["Exterior2nd"]].shape[0]


# In[53]:


1245/data.shape[0]


# In[54]:


#more than 85 % information are the same we can delete Exterior2nd


# In[55]:


data= data.drop("Exterior2nd",axis=1)


# In[56]:


data["Exterior1st"].value_counts()


# In[57]:


fig = sns.FacetGrid(data,hue='Exterior1st',aspect=4,legend_out=False)
fig.map(sns.kdeplot,'log_SalePrice',shade=True)

upper = data['log_SalePrice'].max()
lower = data['log_SalePrice'].min()
fig.set(xlim=(lower,upper))

fig.add_legend()


# In[58]:


# based on the value-counts VinylSd  HdBoard  MetalSd   Wd Sdng are the top 4 that contain most information. 
# as for these four types only VinylSd has a huge difference from others. so we will set as VinylSd and other


# In[59]:


data["Exterior1st"]=pd.get_dummies(data["Exterior1st"])["VinylSd"]


# In[60]:


data["MasVnrType"].value_counts()


# In[61]:


fig = sns.FacetGrid(data,hue='MasVnrType',aspect=4,legend_out=False)
fig.map(sns.kdeplot,'log_SalePrice',shade=True)

upper = data['log_SalePrice'].max()
lower = data['log_SalePrice'].min()
fig.set(xlim=(lower,upper))

fig.add_legend()


# In[62]:


# based on the graph, None and BrkCmn have similar pattern


# In[63]:


MasVnrType_dummy=pd.get_dummies(data["MasVnrType"],prefix="MasVnrType")[["MasVnrType_BrkFace","MasVnrType_Stone"]]


# In[64]:


data.drop("MasVnrType",axis=1,inplace=True)


# In[65]:


data= pd.concat([data,MasVnrType_dummy],axis=1)


# In[66]:


data["Foundation"].value_counts()


# In[67]:


fig = sns.FacetGrid(data,hue='Foundation',aspect=4,legend_out=False)
fig.map(sns.kdeplot,'log_SalePrice',shade=True)

upper = data['log_SalePrice'].max()
lower = data['log_SalePrice'].min()
fig.set(xlim=(lower,upper))

fig.add_legend()


# In[68]:


# top two category already conver mist information, so we can just select these two 


# In[69]:


Foundation_dummy=pd.get_dummies(data["Foundation"],prefix="Foundation")[["Foundation_PConc","Foundation_CBlock"]]


# In[70]:


data=data.drop("Foundation",axis=1)


# In[71]:


data= pd.concat([data,Foundation_dummy],axis=1)


# In[72]:


data["HouseStyle"].value_counts()


# In[73]:


fig = sns.FacetGrid(data,hue='HouseStyle',aspect=4,legend_out=False)
fig.map(sns.kdeplot,'log_SalePrice',shade=True)

upper = data['log_SalePrice'].max()
lower = data['log_SalePrice'].min()
fig.set(xlim=(lower,upper))

fig.add_legend()


# In[74]:


HouseStyle_dummy=pd.get_dummies(data["HouseStyle"],prefix="HouseStyle")[["HouseStyle_1Story","HouseStyle_2Story"]]


# In[75]:


data=data.drop("HouseStyle",axis=1)


# In[76]:


data=pd.concat([data,HouseStyle_dummy],axis=1)


# In[77]:


# as for the defination of Mssubclass this should be categorical variable. and the dinfination od it has already been cover in the in story and second story 
# so we can delete this column


# In[78]:


data=data.drop("MSSubClass",axis=1)


# In[79]:


data.shape


# In[80]:


sns.heatmap(data.corr().abs())


# In[81]:


data=data.drop("SalePrice",axis=1)


# In[82]:


data.corr().abs()["log_SalePrice"].sort_values()


# In[83]:


data["TotalSF"]=data["1stFlrSF"]+data["2ndFlrSF"]


# In[84]:


data=data.drop(["1stFlrSF","2ndFlrSF"],axis=1)


# In[85]:


data[["TotalSF","GrLivArea"]]


# In[86]:


data=data.drop("GrLivArea",axis=1)


# In[87]:


data.corr().abs()["log_SalePrice"].sort_values()


# In[88]:


data["TotalBath"]=data[["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath"]].sum(axis=1)


# In[89]:


data=data.drop(["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath"],axis=1)


# In[90]:


data.corr().abs()["log_SalePrice"].sort_values()


# In[91]:


data["TotalPorchSF"]=data[["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"]].sum(axis=1)


# In[92]:


data=data.drop(["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"],axis=1)


# In[93]:


data.corr().abs()["log_SalePrice"].sort_values().index


# In[94]:


data=data.fillna(0)


# In[98]:


data=(data-data.min())/(data.max()-data.min())


# In[141]:


data.shape


# In[152]:


data= data.reindex(np.random.permutation(data.index))


# In[153]:


train_max_row = round(data.shape[0] * .8)


# In[154]:


train=data.loc[:train_max_row]


# In[155]:


test =data.loc[train_max_row:]


# In[160]:


y=train[["log_SalePrice"]]


# In[161]:


X= train[x1_index]


# In[158]:


x1_index


# In[162]:


y1=test[["log_SalePrice"]]


# In[163]:


X1= test[x1_index]


# In[99]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV


# In[100]:


lr= LinearRegression()


# In[101]:


kf=KFold(n_splits=5)


# In[164]:


lr_cross_val_score=cross_val_score(lr,X,y,scoring="neg_mean_squared_error", cv= kf)


# In[165]:


lr_cross_val_score


# In[167]:


lr1_cross_val_score=cross_val_score(lr,X,y,scoring="neg_mean_squared_error", cv= kf)


# In[168]:


lr1_cross_val_score
np.mean(lr1_cross_val_score)


# In[172]:


ridge=Ridge()
parameters={"alpha":[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,45,50,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring="neg_mean_squared_error",cv=kf)
ridge_regressor.fit(X,y)


# In[173]:


ridge_regressor.best_params_


# In[174]:


ridge_regressor.best_score_


# In[176]:


lasso=Lasso()
lasso_regressor=GridSearchCV(lasso,parameters,scoring="neg_mean_squared_error",cv=kf)
lasso_regressor.fit(X,y)


# In[177]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[112]:


from sklearn.neighbors import KNeighborsRegressor


# In[179]:


kn= KNeighborsRegressor()
kn_cross_val_score=cross_val_score(kn,X,y,cv=kf,scoring="neg_mean_squared_error")
kn_cross_val_score


# In[180]:


np.mean(kn_cross_val_score)


# In[183]:



result=[]
for k in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]:
    kn= KNeighborsRegressor(n_neighbors=k)
    kn.fit(X,y)
    result_list=cross_val_score(kn,X,y,cv=kf,scoring="neg_mean_squared_error")
    result.append(np.mean(result_list))


# In[184]:


result


# In[185]:


np.max(result)


# In[ ]:


# when k=6


# In[186]:


from sklearn.tree import DecisionTreeRegressor


# In[188]:


tree_result=[]
for n in [5,10,20,30,40,50,60,70,80,90,100,110]:
    dt = DecisionTreeRegressor( min_samples_leaf=n)
    dt_cross_val_score=cross_val_score(dt,X,y,cv=kf,scoring="neg_mean_squared_error")
    tree_result.append(np.mean(dt_cross_val_score))


# In[189]:


tree_result


# In[192]:


tree_result=[]
for n in [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
    dt = DecisionTreeRegressor( min_samples_leaf=n)
    dt_cross_val_score=cross_val_score(dt,X,y,cv=kf,scoring="neg_mean_squared_error")
    tree_result.append(np.mean(dt_cross_val_score))


# In[193]:


tree_result


# In[194]:


np.max(tree_result)


# In[133]:


# when we set the leaves as 12, we can get the best result the rmse is 0.03260972314659545


# In[195]:


from sklearn.ensemble import RandomForestRegressor


# In[196]:


rf_result=[]
for i in [10,20,30,40,50,60]:
    rf= RandomForestRegressor(min_samples_leaf=i)
    result =cross_val_score(rf,X,y,cv=kf,scoring="neg_mean_squared_error")
    rf_result.append(np.mean(result))


# In[197]:


result


# In[204]:


rf_result=[]
for i in range(10,30):
    rf= RandomForestRegressor(min_samples_leaf=i)
    result =cross_val_score(rf,X,y,cv=kf,scoring="neg_mean_squared_error")
    rf_result.append(np.mean(result))


# In[205]:


rf_result


# In[206]:


np.max(rf_result)


# In[207]:


# when we set the leave as 15 we can get the best model based on the randomforest model, the smallest rmse is 0.03225307358458143


# In[208]:


rf= RandomForestRegressor(min_samples_leaf=15)


# In[209]:


rf.fit(X,y)


# In[211]:


prediction=rf.predict(X1)


# In[213]:


from sklearn.metrics import mean_squared_error


# In[217]:


mse=mean_squared_error(y1,prediction)


# In[216]:


x1_index


# In[218]:


np.sqrt(mse)


# In[ ]:




