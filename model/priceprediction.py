import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams["figure.figsize"]=(20,10)

df1=pd.read_csv(r"D:\ai  waste classifier\ML Concepts\RealEstatePricePrediction\model\bengaluru_house_prices.csv")
# print(df1.head())
# print(df1.shape)
# print(df1.columns)
# print(df1["area_type"].unique())
# print(df1["area_type"].value_counts())

# drop the features that are not required to build model

df2=df1.drop(["area_type","society","balcony","availability"],axis="columns")
# print(df2.shape)

# data cleaning / handling NA values
# print(df2.isnull().sum())
df3=df2.dropna().copy()
# print(df3.isnull().sum())
# print(df3.shape)

# feature engineering

df3["bhk"]=df3['size'].apply(lambda x: int(x.split(' ')[0]))
# print(df3.bhk.unique())


# Above shows that total_sqft can be a range (e.g. 2100-2850). For such case we can just take average of min and max value in the range

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


df3[~df3['total_sqft'].apply(is_float)].head(10)

def convert_sqft_to_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return(float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    
df4=df3.copy()
df4.total_sqft=df4.total_sqft.apply(convert_sqft_to_num)
df4=df4[df4.total_sqft.notnull()]
# print(df4['total_sqft'].head())
# print(df4.loc[30])

# Add new feature called price per square feet

df5=df4.copy()
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
# print(df5.head())

df5_stats=df5["price_per_sqft"].describe()
# print(df5_stats)

# df5.to_csv("bhp.csv",index=False)

df5.location=df5.location.apply(lambda x: x.strip())
location_stats=df5['location'].value_counts(ascending=False)
# print(location_stats)

# print(location_stats.values.sum())
# print(len(location_stats[location_stats>10]))

# dimensionality reduction

location_stats_less_than_10=location_stats[location_stats<10]
# print(location_stats_less_than_10)

df5.location=df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
# print(df5.head(10))


# Outlier Removal Using Business Logic

'''As a data scientist when you have a conversation with your business manager
 (who has expertise in real estate), 
 he will tell you that normally square ft per bedroom is 300 
 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 400 sqft apartment 
 with 2 bhk than that seems suspicious and can be removed as an outlier. We will remove such 
 outliers by keeping our minimum thresold per bhk to be 300 sqft'''

# print(df5[df5.total_sqft/df5.bhk<300].head())

df6=df5[~(df5.total_sqft/df5.bhk<300)]
# print(df6.shape)
# print(df6.price_per_sqft.describe())


# min price per sqft is 267 rs/sqft whereas max is 12000000
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7=remove_pps_outliers(df6)

# print(df7.shape)

#  2 BHK and 3 BHK property prices look like

def plot_scatter_chart(df, location):
    bhk2=df[(df.location==location) & (df.bhk==2)]
    bhk3=df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 bhk', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green',label='3 bhk', s=50)
    plt.xlabel("Total sqft")
    plt.ylabel("Price(Lakh INR)")
    plt.title(location)
    plt.legend()
    plt.show()

# plot_scatter_chart(df7,"Rajaji Nagar")
# plot_scatter_chart(df7,"Hebbal")

def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }

        for bhk, bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8=remove_bhk_outliers(df7)

# print(df8.shape)

# plot_scatter_chart(df8,"Rajaji Nagar")
# plot_scatter_chart(df8,"Hebbal")


# import matplotlib
# matplotlib.rcParams["figure.figsize"] = (20,10)
# plt.hist(df8.price_per_sqft,rwidth=0.8)
# plt.xlabel("Price Per Square Feet")
# plt.ylabel("Count")
# plt.show()

# Outlier Removal Using Bathrooms Feature

# print(df8.bath.unique())

# plt.hist(df8.bath,rwidth=0.8)
# plt.xlabel("Numbers of Bathrooms")
# plt.ylabel("Count")
# plt.show()


# print(df8[df8.bath>10])

# It is unusual to have 2 more bathrooms than number of bedrooms in a home


# print(df8[df8.bath>df8.bhk+2])

df9=df8[df8.bath<df8.bhk+2]
# print(df9.shape)

# print(df9.head())


df10=df9.drop(['size','price_per_sqft'],axis='columns')
# print(df10.head())

# one hot encoding
dummies=pd.get_dummies(df10.location,dtype=int)
# print(dummies.head())

df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
# print(df11.head())

df12=df11.drop('location',axis='columns')
# print(df12.head())


# print(df12.shape)

# build a model now
x=df12.drop(['price'],axis="columns")
# print(x.head())
# print(x.shape)
y=df12.price
# print(y.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression

lr_clf=LinearRegression()
lr_clf.fit(x_train,y_train)
# print(lr_clf.score(x_test,y_test))

# Use K Fold cross validation to measure accuracy of our LinearRegression model

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# print(cross_val_score(LinearRegression(),x,y,cv=cv))

# Find best model using GridSearchCV

# from sklearn.model_selection import GridSearchCV, ShuffleSplit
# from sklearn.linear_model import LinearRegression, Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler


# def find_best_model_using_gridsearchcv(x, y):
#     algos = {
#         'linear_regression': {
#             'model': Pipeline([
#                 ('scaler', StandardScaler(with_mean=False)),
#                 ('lr', LinearRegression())
#             ]),
#             'params': {
#                 'lr__fit_intercept': [True, False]
#             }
#         },
#         'lasso': {
#             'model': Pipeline([
#                 ('scaler', StandardScaler(with_mean=False)),
#                 ('lasso', Lasso())
#             ]),
#             'params': {
#                 'lasso__alpha': [0.1, 1, 10],
#                 'lasso__selection': ['random', 'cyclic']
#             }
#         },
#         'decision_tree': {
#             'model': DecisionTreeRegressor(),
#             'params': {
#                 'criterion': ['squared_error', 'friedman_mse'],
#                 'splitter': ['best', 'random']
#             }
#         }
#     }

#     scores = []
#     cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

#     for algo_name, config in algos.items():
#         gs = GridSearchCV(config['model'], config['params'],
#                           cv=cv, scoring='r2', return_train_score=False)
#         gs.fit(x, y)
#         scores.append({
#             'model': algo_name,
#             'best_score': gs.best_score_,
#             'best_params': gs.best_params_
#         })

#     return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# print(find_best_model_using_gridsearchcv(x,y))

def predict_price(location, sqft, bath, bhk):
    import pandas as pd
    x_input = pd.DataFrame(columns=x.columns)
    x_input.loc[0] = 0
    x_input.at[0, 'total_sqft'] = sqft
    x_input.at[0, 'bath'] = bath
    x_input.at[0, 'bhk'] = bhk
    if location in x.columns:
        x_input.at[0, location] = 1
    return lr_clf.predict(x_input)[0]


# print(predict_price('1st Phase JP Nagar',1000,2,2))
# print(predict_price('1st Phase JP Nagar',1000,3,3))

# print(predict_price('Indira Nagar',1000,2,2))
# print(predict_price('Indira Nagar',1000,3,3))


# Export the tested model to a pickle file
import pickle

with open('banglore_home_price_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

# Export location and column information to a file that will be useful later on in our prediction application

import json
columns={
        'data_columns':[col.lower() for col in x.columns]

    }
with open("columns.json",'w')as f:
    f.write(json.dumps(columns))