#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("Gold_price_data (4).csv")


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df = df.drop(['total_cases'], axis=1)


# In[6]:


df.tail(12)


# In[7]:


df.shape


# In[8]:


df = df.fillna(0)


# In[9]:


df.tail(10)


# In[10]:


df.info()


# In[11]:


corr = df.corr()


# In[12]:


corr["Gold_prices"].sort_values(ascending=False)


# In[13]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df.hist(bins=50, figsize=(24,16))
plt.show()


# In[14]:


fig = plt.figure(figsize=(26,18))
cor_matrix = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True, fmt='.2f', linewidth=.9, cmap=sns.diverging_palette(20, 220, n=256))
cor_matrix.set_xticklabels( cor_matrix.get_xticklabels(), rotation=30,  horizontalalignment='right', fontweight='light' )
plt.show()


# In[15]:


df = df.drop(['US_bonds', 'XOM','FTSE_100','Natural_gas'],axis=1)


# In[16]:


df = df.drop(['UK_bonds','Crude_oil','cocoa_london_price',],axis=1)


# In[17]:


df["SPDR_GLD"].plot(figsize=(10, 7),color='r')
plt.ylabel("Gold ETF Prices")
plt.title("Gold ETF Price Series")
plt.show()


# In[18]:


corr["Gold_prices"].sort_values(ascending=False)


# In[19]:


df['Covid_log2'] = np.log2(df['COVID_cases'])


# In[20]:


df.head()


# In[21]:


sns.distplot(df['Gold_prices'],color='gold')


# In[22]:


df.plot(kind="scatter", x="Covid_log2", y="Gold_prices")


# In[23]:


df['Bitcoin_log2'] = np.log2(df['Bitcoin'])


# In[24]:


df.plot(kind="scatter",x="Bitcoin_log2",y="Gold_prices")


# In[25]:


df = df.drop(['SPDR_GLD','Copper','Platinum_price','Palladium'],axis=1)


# In[26]:


df = df.drop(['Euro_stoxx_50','Bitcoin','COVID_cases','Date'],axis=1)


# In[27]:


df.head(2)


# In[28]:


df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]


# In[29]:


train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)


# In[30]:


test_set.head()


# In[31]:


df = train_set.copy()


# In[32]:


df_labels = df["Gold_prices"]


# In[33]:


df = df.drop("Gold_prices", axis=1) # drop labels for training set


# In[35]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
std_scaler = preprocessing.StandardScaler()

numerical_attribs = list(df)
df[numerical_attribs] = std_scaler.fit_transform(df[numerical_attribs])

df.head(10)


# In[36]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(df, df_labels)


# In[37]:


from sklearn.metrics import mean_squared_error

df_predictions = lin_reg.predict(df)
lin_mse = mean_squared_error(df_labels, df_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
print('Coefficients: ', lin_reg.coef_)
print(lin_reg.intercept_)


# In[38]:


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(df_labels, df_predictions)
lin_mae


# # Cross Validation

# In[42]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, df, df_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[44]:


lin_scores = cross_val_score(lin_reg, df, df_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(lin_rmse_scores)


# #### Random forest regressor

# In[45]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(df, df_labels)


# In[46]:


df_predictions = forest_reg.predict(df)
forest_mse = mean_squared_error(df_labels, df_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# ### cross validation score

# In[48]:


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, df, df_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print(forest_rmse_scores)


# In[49]:


scores = cross_val_score(lin_reg, df, df_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()


# In[50]:


from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(df, df_labels)
df_predictions = svm_reg.predict(df)
svm_mse = mean_squared_error(df_labels, df_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse


# **Lasso Regression**

# In[51]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
model = Lasso()
X, y = df, df_labels

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Fitting the model
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Model Predictions
from sklearn.metrics import mean_absolute_error,r2_score
print(r2_score(y_test,y_pred))


# In[52]:


#Visualizing the results
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.distplot(y_test-y_pred)


# **Hyper Parameter Tuning for Lasso Regression in Python**

# In[53]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
#define parameters

param = {
    'alpha':[.00001, 0.0001,0.001, 0.01],
    'fit_intercept':[True,False],
    'normalize':[True,False],
    'positive':[True,False],
    'selection':['cyclic','random'],
    }
# define search
search = GridSearchCV(model, param, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)

# execute search
result = search.fit(X_sc, y)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


# In[54]:


#fitting these hyperparameters into model
model = Lasso(alpha=0.001,fit_intercept= True, normalize = True, positive= False, selection = 'cyclic')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(r2_score(y_test,y_pred))


# # Hyperparameter tuning

# ## Grid search

# In[55]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(df, df_labels)


# In[56]:


grid_search.best_params_


# In[57]:


grid_search.best_estimator_


# In[58]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[59]:


pd.DataFrame(grid_search.cv_results_)


# In[60]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[62]:


attributes = numerical_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[63]:


final_model = grid_search.best_estimator_

X_test = test_set.drop("Gold_prices", axis=1)
y_test = test_set["Gold_prices"].copy()
num_attribs = list(X_test)
X_test[num_attribs] = std_scaler.fit_transform(X_test[num_attribs])


final_predictions = final_model.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[64]:


final_rmse


# In[65]:


from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))


# In[66]:


m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)


# In[67]:


zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)


# In[68]:


Y_test = list(df_labels)
plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(df_predictions, color='green', label='final_predictions')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[ ]:





# # LSTM Reccurent neural network model

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import LSTM


# In[69]:


df1 = pd.read_excel('LBMA_gold.xlsx')


# In[70]:


df1.head()


# In[71]:


df1.info()


# In[72]:


df1.isnull().sum()


# In[73]:


df1['DATE'] = pd.to_datetime(df1['DATE'])
df1.sort_values(by='DATE',ascending=True, inplace=True)
df1.reset_index(drop=True,inplace=True)


# In[74]:


df1.head()


# In[75]:


df1['PM FIX USD ($)'] = pd. to_numeric(df1['PM FIX USD ($)']) 


# In[76]:


df1['open'] = df1['AM FIX USD ($)']


# In[77]:


df1['close'] = df1['PM FIX USD ($)']


# In[78]:


df1.info()


# In[79]:


df1.drop(['AM FIX USD ($)','PM FIX USD ($)'],axis=1, inplace = True)


# In[80]:


df1.duplicated().sum()


# In[81]:


df1 = df1.fillna(df1.mean())


# In[82]:


test_size = df1[df1.DATE.dt.year==2022].shape[0]
test_size


# In[83]:


plt.figure(figsize=(15,6), dpi=150)
plt.rcParams['axes.facecolor']= 'yellow'
plt.rc('axes',edgecolor='white')
plt.plot(df1.DATE[:-test_size], df1.close[:-test_size],color='black',lw=2)
plt.plot(df1.DATE[-test_size:], df1.close[-test_size:],color='red',lw=2)
plt.title('gold price train and test', fontsize = 15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('price', fontsize=12 )
plt.legend(['train set','test set'], loc='upper left', prop={'size':15})
plt.grid(color='white')
plt.savefig("train_test.png")
plt.show()


# In[84]:


scaler = MinMaxScaler()
scaler.fit(df1.close.values.reshape(-1,1))


# In[85]:


window_size = 60


# In[86]:


train_data = df1.close[:-test_size]
train_data = scaler.transform(train_data.values.reshape(-1,1))


# In[87]:


X_train = []
y_train = []

for i in range(window_size, len(train_data)):
    X_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])


# In[88]:


test_data = df1.close[-test_size-60:]
test_data = scaler.transform(test_data.values.reshape(-1,1))


# In[89]:


X_test = []
y_test = []

for i in range(window_size, len(test_data)):
    X_test.append(test_data[i-60:i,0])
    y_test.append(test_data[i,0])
    


# In[90]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[91]:


X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))
y_train = np.reshape(y_train,(-1,1))
y_test = np.reshape(y_test,(-1,1))  


# In[92]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[93]:


def define_model():
    input1 = Input(shape=(window_size,1))
    x = LSTM (units=64, return_sequences = True)(input1)
    x = Dropout(0.2)(x)
    x = LSTM(units = 64, return_sequences = True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units = 64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation = 'softmax')(x)
    
    dnn_output = Dense(1)(x)
    
    model = Model(inputs= input1, outputs = [dnn_output])
    model.compile(loss = 'mean_squared_error', optimizer = 'Nadam')
    model.summary()
    
    return model


# In[ ]:


model = define_model()
history = model.fit(X_train, y_train, epochs = 150, batch_size = 32, validation_split = 0.1, verbose = 1)


# In[ ]:


result = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)


# In[ ]:


MAPE = mean_absolute_percentage_error(y_test, y_pred)
accuracy = 1 - MAPE


# In[ ]:


print(result)
print(MAPE)
print(accuracy)


# In[ ]:


y_test_true = scaler.inverse_transform(y_test)
y_test_pred = scaler.inverse_transform(y_pred)


# In[ ]:


plt.figure(figsize=(15,6), dpi=150)
plt.rcParams['axes.facecolor']= 'white'
plt.rc('axes',edgecolor='white')
plt.plot(df1['DATE'].iloc[:-test_size], scaler.inverse_transform(train_data),color='blue',lw=2)
plt.plot(df1['DATE'].iloc[-test_size:], y_test_true,color='green',lw=2)
plt.plot(df1['DATE'].iloc[-test_size:], y_test_pred,color='red',lw=2)
plt.title('gold price train and test', fontsize = 15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('price', fontsize=12 )
plt.legend(['Training Data','Actual Test Data','Predicted Test Data'], loc='upper left', prop={'size':15})
plt.grid(color='orange')
plt.savefig("prediction_graph.png")
plt.show()


# In[ ]:


last_day = test_data[-window_size:]


# In[ ]:


next_day = model.predict(last_day.reshape(1,window_size,1))
next_day = scaler.inverse_transform(next_day)[0][0]


# In[ ]:


print(next_day)

