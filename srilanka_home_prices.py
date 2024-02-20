#!/usr/bin/env python
# coding: utf-8

# <h1 style='color:purple' align='center'>Data Science Regression Project: Predicting Home Prices in Sri Lanka</h1>

# In[25]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# <h2 style='color:blue'>Data Load: Load Sri Lanka home prices into a dataframe</h2>

# In[26]:


csv_file_path = "C:\\Users\\ASUS\\Desktop\\cc\\new\\house_prices.csv"
df1 = pd.read_csv(csv_file_path)
df1.head()


# In[27]:


df1.shape


# In[28]:


df1.columns


# In[29]:


df1['Location'].unique()


# In[ ]:


# df1['Location'].value_counts()


# **Drop features that are not required to build our model**

# In[31]:


df2 = df1.drop(['Title','Seller_type','Post_URL','Sub_title','Address','Description','Seller_name','published_date','Geo_Address','Lat','Lon'],axis='columns')
df2.shape
df2.head()


# df2<h2 style='color:blue'>Data Cleaning: Handle NA values</h2>

# In[32]:


df2.isnull().sum()


# In[33]:


df2.shape


# <h2 style='color:blue'>Feature Engineering</h2>

# **Add new feature(integer) for bhk (Bedrooms Hall Kitchen)**

# **Explore total_sqft feature**

# In[34]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[35]:


df2['land_size_perch'] =df2['Land size'].apply(lambda x: x.split(' ')[0])
df2['house_size_sqft'] =df2['House size'].apply(lambda x: x.split(' ')[0])
df2['price_RS'] =df2['Price'].apply(lambda x: x.split(' ')[1])
df3=df2.drop(['Land size','House size', 'Price'],axis='columns')
df3.head()


# In[36]:


df3[~df3['price_RS'].apply(is_float)].head()


# In[37]:


df3.loc[100]


# <h2 style="color:blue">Feature Engineering</h2>

# **Add new feature called price per square feet**

# In[64]:


df3['land_size_perch'] = df3['land_size_perch'].astype(str).str.replace(',', '').astype(float)
df3['land_size_perch'] = pd.to_numeric(df3['land_size_perch'])
df3['land_size_perch'].head(15)
df4=df3.copy()


# In[65]:


df4['price_RS'] = df4['price_RS'].astype(str).str.replace(',', '').astype(float)
df4['price_RS'] = pd.to_numeric(df4['price_RS'])
df4['price_RS'].head(15)


# In[124]:


data_no_zeros = np.where(df4 == 0, np.nan, df4)
data_no_zeros = df4.replace(0, float('nan'))
print(data_no_zeros)


# In[125]:


zero_columns = data_no_zeros.columns[data_no_zeros.eq(0).any()]
print("Columns with zeros:", zero_columns)
df4=data_no_zeros.copy()
df4


# In[126]:


df4['price_in_laks'] = df4['price_RS']/100000
df4 = df4[df4['house_size_sqft'] != '0']
df4['house_size_sqft'] = df4['house_size_sqft'].astype(str).str.replace(',', '').astype(float)
df4['house_size_perch'] = df4['house_size_sqft']/272.25
df4['price_per_perch_in_laks']=df4['price_in_laks']/df4['house_size_perch']
df5=df4.drop(['house_size_sqft','price_RS' ], axis='columns')
df5.head()


# In[144]:


df6_stats = df5['price_per_perch_in_laks'].describe()
df6_stats


# ##### df6.to_csv("shp.csv",index=False)

# **Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations**

# In[149]:


extract_first_word = lambda sentence: sentence.split(',')[0].strip()
df5['Location'] = df5['Location'].apply(extract_first_word)
location_stats = df5['Location'].value_counts(ascending=False)
location_stats


# In[150]:


location_stats.values.sum()


# In[151]:


len(location_stats[location_stats>10])


# In[152]:


len(location_stats)


# In[153]:


len(location_stats[location_stats<=10])


# <h2 style="color:blue">Dimensionality Reduction</h2>

# **Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, it will help us with having fewer dummy columns**

# In[154]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[155]:


len(df5.Location.unique())


# In[156]:


df5.Location = df5.Location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.Location.unique())


# In[162]:


df5.head(5)
df6=df5.copy()
df6


# <h2 style="color:blue">Outlier Removal Using Business Logic</h2>

# **In the sense of business manager (who has expertise in real estate), they will know that normally square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier. We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft**

# In[164]:


df6 = df6[df6['Beds'] != '10+']
df6['Beds'] = df6['Beds'].astype(str).str.replace(',', '').astype(float)
df6[df6.house_size_perch/df6.Beds<(300/272.25)].head()


# **Check above data points. We have 6 bhk apartment with 1020 sqft. Another one is 8 bhk and total sqft is 600. These are clear data errors that can be removed safely**

# In[165]:


df6.shape


# In[166]:


df7 = df6[~(df6.house_size_perch/df6.Beds<(300/272.25))]
df7.shape


# <h2 style='color:blue'>Outlier Removal Using Standard Deviation and Mean</h2>

# In[167]:


df7.price_per_perch_in_laks.describe()


# **Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices. We should remove outliers per location using mean and one standard deviation**

# In[168]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df7.groupby('Location'):
        m = np.mean(subdf.price_per_perch_in_laks)
        st = np.std(subdf.price_per_perch_in_laks)
        reduced_df = subdf[(subdf.price_per_perch_in_laks>(m-st)) & (subdf.price_per_perch_in_laks<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df8 = remove_pps_outliers(df7)    
df8.shape


# **Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like**

# In[170]:


def plot_scatter_chart(df,Location):
    beds2 = df8[(df8.Location==Location) & (df8.Beds==2)]
    beds3 = df8[(df8.Location==Location) & (df8.Beds==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(beds2.house_size_perch,beds2.price_per_perch_in_laks,color='blue',label='2 BHK', s=50)
    plt.scatter(beds3.house_size_perch,beds3.price_per_perch_in_laks,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total perch Area")
    plt.ylabel("Price (Sri Lankan Rupees)")
    plt.title(Location)
    plt.legend()
    
    
    
plot_scatter_chart(df8,"Kadawatha")


# In[171]:


plot_scatter_chart(df8,"Dehiwala")


# **We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per bhk, i.e.**
# ```
# {
#     '1' : {
#         'mean': 4000,
#         'std: 2000,
#         'count': 34
#     },
#     '2' : {
#         'mean': 4300,
#         'std: 2300,
#         'count': 22
#     },    
# }
# ```
# **Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment**

# In[172]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('Location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('Beds'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_perch_in_laks),
                'std': np.std(bhk_df.price_per_perch_in_laks),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('Beds'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_perch_in_laks<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df9 = remove_bhk_outliers(df8)
df9.shape


# **Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties**

# In[173]:


plot_scatter_chart(df9,"Kadawatha")


# In[174]:


plot_scatter_chart(df9,"Dehiwala")


# **Based on above charts we can see that data points highlighted in red below are outliers and they are being removed due to remove_bhk_outliers function**

# In[176]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df9.price_per_perch_in_laks,rwidth=0.8)
plt.xlabel("price_per_perch_in_laks")
plt.ylabel("Count")


# <h2 style='color:blue'>Outlier Removal Using Bathrooms Feature</h2>

# In[177]:


df9.Baths.unique()


# In[178]:


plt.hist(df9.Baths,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[191]:


df9['Baths'] = df9['Baths'].astype(str).str.replace(',', '').astype(float)
bath_details = df9[df9.Baths>10]
bath_details


# **It is unusual to have 2 more bathrooms than number of bedrooms in a home**

# In[190]:


df9[df9.Baths>df9.Beds+2]
bath_stats = df9['Baths'].value_counts(ascending=False)
bath_stats


# **According to the business logic that if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error and can be removed**

# In[192]:


df10 = df9[df9.Baths<df9.Beds+2]
df10.shape


# In[193]:


df10.head(5)


# In[194]:


df11 = df10.drop(['land_size_perch','price_per_perch_in_laks'],axis='columns')
df11.head(3)


# <h2 style='color:blue'>Use One Hot Encoding For Location</h2>

# In[195]:


dummies = pd.get_dummies(df11.Location)
dummies.head(5)


# In[199]:


df12 = pd.concat([df11,dummies.drop('other',axis='columns')],axis='columns')
df12.head(7)


# In[202]:


df13 = df12.drop('Location',axis='columns')
df13.head(5)  


# <h2 style='color:blue'>Build a Model Now...</h2>

# In[203]:


df13.shape


# In[204]:


X = df13.drop(['price_in_laks'],axis='columns')
X.head(3)


# In[205]:


X.shape


# In[208]:


y = df13.price_in_laks
y.head(5)


# In[209]:


len(y)


# In[210]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=13)


# In[262]:


from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)
reg.score(X_test,y_test)


# In[263]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# <h2 style='color:blue'>Use K Fold cross validation to measure accuracy of our LinearRegression model</h2>

# In[219]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=6, test_size=0.2, random_state=10)

cross_val_score(LinearRegression(), X, y, cv=cv)


# <h2 style='color:blue'>Use K Fold cross validation to measure accuracy of our Decision Tree model</h2>

# In[277]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=6, test_size=0.2, random_state=0)

cross_val_score(DecisionTreeRegressor(), X, y, cv=cv)


# <h2 style='color:blue'>Find best model using GridSearchCV</h2>

# In[278]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# **Based on above results we can say that Decision gives the best score. Hence we will use that.**

# <h2 style='color:blue'>Test the model for few properties</h2>

# In[279]:


def predict_price(Location,house_size_perch,Baths,Beds):    
    loc_index = np.where(X.columns==Location)[0][0]

    x = np.zeros(len(X.columns))
    x[2] = house_size_perch
    x[0] = Baths
    x[1] = Beds
    if loc_index >= 0:
        x[loc_index] = 1

    return reg.predict([x])[0] 


# In[280]:


predict_price('Angoda',5, 2, 2)


# In[284]:


predict_price('Angoda',7, 2, 3)


# In[282]:


predict_price('Angoda',6, 2, 2)


# In[258]:


predict_price('Angoda',10, 2, 2)


# <h2 style='color:blue'>Export the tested model to a pickle file</h2>

# In[285]:


import pickle
with open('sri_lanka_home_price_prediction.pickle','wb') as f:
    pickle.dump(reg,f)


# <h2 style='color:blue'>Export location and column information to a file that will be useful later on in our prediction application</h2>

# In[286]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




