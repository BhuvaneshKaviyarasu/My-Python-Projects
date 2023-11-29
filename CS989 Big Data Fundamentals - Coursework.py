#!/usr/bin/env python
# coding: utf-8

# - <h2> Import Packages

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from sklearn import preprocessing  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings


# - <h2> Import the Dataset with the Dataframe name "pythondata"

# In[3]:


music_df=pd.read_csv('C:\\Users\\bhuvanesh\\Documents\\python dataset\\pythondata.csv')


# - <h4> Get first five values from the Dataset

# In[4]:


music_df.head()


# - <h4> Get the Shape of the Dataset
#    

# In[5]:


music_df.shape


# - <h4> To get the Information about the dataset

# In[6]:


music_df.info


# - <h4> To get the Data Types of the Dataset 

# In[7]:


music_df.dtypes


# - <h4> To find whether the Dataset has duplicates 

# In[8]:


music_df.duplicated().any()


# - <h4> To get the Duplicate Values

# In[9]:


duplicates=music_df.duplicated()
music_df[duplicates]


# - <h2> Cleaning the Data:
#   <h4> Removing the Null Values
#   <h4> Dropping the Empty Fields in the Column ("Artist_name")
#   <h4> Dropping "?" in the Column ("Tempo")

# In[10]:


music_df.drop([10000, 10001, 10002, 10003, 10004], inplace = True)
music_df = music_df.drop(music_df[music_df["artist_name"] == "empty_field"].index)
music_df = music_df.drop(music_df[music_df["tempo"] == "?"].index)


# - <h4> To get the shape of the Dataset
#    

# In[11]:


music_df.shape


# - <h4> Removing the High-entropy values and checking the number of null values:

# In[12]:


music_df=music_df.drop(["instance_id", "obtained_date", "artist_name"], axis=1)
pd.isnull(music_df).sum().sum()


# - <h4> Sorting top 10 values by "Popularity"

# In[13]:


dff=music_df.sort_values(by=['popularity'], ascending=False).head(10)
dff.nlargest(n=10, columns=['popularity'])
dff


# - <h4> Finding the top 10 popular songs:
# 

# In[14]:


dff.plot.barh(x='track_name', y='popularity', color='green', fill = True)
plt.title('Top 10 Popular songs')
plt.ylabel('Name of the Tracks ')
plt.xlabel('Popularity')


# - <h4> Removing High-Entropy Values:
#   <h4> Dropping Column "Track_name" from the Dataset 

# In[15]:


music_df=music_df.drop(["track_name"], axis=1)
music_df.head()


# - <h4> Getting the Unique Values from the Dataset

# In[16]:


print(music_df.nunique())


# - <h4> Creating Heatmap after cleaning the data.

# In[17]:


plt.figure(figsize=(8, 8))
sns.heatmap(music_df.corr(), linewidth=1, cbar=True, cmap='RdBu', annot=False)


# - <h4> Plotting "Number of songs in each key"

# In[18]:


def plot_counts(feature, order = None):
    sns.countplot(x = feature, data = music_df, palette = "ocean", order = order)
    plt.title("Number of songs in each category")
    plt.show()
plot_counts("key", ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])


# - <h4> Plotting "The number of songs in Mode"

# In[19]:


plot_counts("mode")


# - <h4> Plotting "the number of songs in music_genre"

# In[20]:


plt.figure(figsize = (9, 4))
plot_counts("music_genre")


#   <h4> Converting tempo as float value
#   <h4> Creating a new Dataframe "Numerical_Values", adding all the Columns except key, music_genre, mode. 
#   <h4> This Numerical_Values is to store all the numericals in one place

# In[21]:


music_df["tempo"] = music_df["tempo"].astype("float")
music_df["tempo"] = np.around(music_df["tempo"], decimals = 2)
Numerical_Values = music_df.drop(["key", "music_genre", "mode"], axis = 1)


# - <h2> Creating Subplots for all the Columns in Numerical_Values Column

# In[22]:


fig, axs = plt.subplots(ncols = 3, nrows = 4, figsize = (15, 15))
fig.delaxes(axs[3][2])
index = 0

axs = axs.flatten()
for k, v in Numerical_Values.items():
    sns.histplot(v, ax = axs[index])
    index += 1
plt.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 5.0)


# - <h4> Grouping "music_genre" with all other columns in "Numerical_Values"

# In[23]:


for feature in Numerical_Values:
    display(music_df[[feature, 'music_genre']].groupby(['music_genre'], 
 as_index=False).mean().sort_values(by=feature, ascending=False))


# - <h4> Grouping all the "Numerical_Values" with Key, Mode and music_genre as mean values

# In[24]:


for column in music_df.select_dtypes(include='object'):
    display(music_df.groupby(column).mean())


# - <h4> Creating the Heatmap after co-relating the datas

# In[25]:


plt.figure(figsize=(10, 10))
sns.heatmap(music_df.corr(), cmap='RdBu_r', annot=False)


# - <h2> Creating Boxplots for all Numerical Values

# In[26]:


Numerical_Values.describe()


# In[27]:


fig, axs = plt.subplots(ncols = 2, nrows = 6, figsize = (25, 35))
idx = 0
axs = axs.flatten()
for k, v in Numerical_Values.items():
    sns.boxplot(y = k, data = Numerical_Values, ax = axs[idx])
    idx += 1
plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 5.0)


# - <h4> Creating Graphs for key, mode and music_genre based on music_genre

# In[31]:


for column in music_df.select_dtypes(include='object'):
    sns.catplot(x='music_genre', hue=column, data=music_df, kind='count', aspect=4)    
    plt.xticks(rotation=45)
    sns.set(font_scale=2)
    g.set_title(column)


# - <h2> Encoding the (key, mode, music_genre)

# In[32]:


key_encoder = LabelEncoder()


# In[33]:


music_df["key"] = key_encoder.fit_transform(music_df["key"])


# In[34]:


music_df.head()


# In[35]:


key_encoder.classes_


# In[36]:


mode_encoder = LabelEncoder()


# In[37]:


music_df["mode"] = mode_encoder.fit_transform(music_df["mode"])


# In[38]:


music_df['music_genre'].unique()


# In[39]:


music_df.head()


# In[40]:


mode_encoder.classes_


# In[41]:


music_df["music_genre"] = mode_encoder.fit_transform(music_df["music_genre"])


# In[42]:


music_df['music_genre'].unique()


# In[43]:


music_df.head()


# - <h2> Scaling 

# In[44]:


music_features = music_df.drop("music_genre", axis = 1)
music_labels = music_df["music_genre"]


# In[45]:


scaler = StandardScaler()


# In[46]:


Featured_scalar_data = scaler.fit_transform(music_features)


# In[47]:


Featured_scalar_data.mean(), Featured_scalar_data.std()


# In[48]:


sns.kdeplot(Featured_scalar_data[:,5],fill=True, color = 'Green')
plt.xlabel('standardized values')
plt.show()


# - <h2> Training the Dataset

# In[49]:


music_features_train, music_features_test, music_labels_train, music_labels_test = train_test_split(Featured_scalar_data, music_labels, test_size = 0.1, stratify = music_labels)


# In[50]:


train_features, val_features, train_labels, val_labels = train_test_split(music_features_train, music_labels_train, test_size = len(music_labels_test), stratify = music_labels_train)


# - <h2> Logistic Regression

# In[60]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 

logreg = SGDClassifier()
logreg.fit(music_features_train, music_labels_train)
y_pred_train = logreg.predict(music_features_train)
y_pred_test = logreg.predict(music_features_test)
acc=accuracy_score(music_labels_train, y_pred_train)
acc1=accuracy_score(music_labels_test, y_pred_test)
print('Train accuracy is: '+ format(acc))
print('\n')
print('Test accuracy is: '+ format(acc1))
print('\n')
print(classification_report(music_labels_test, y_pred_test))


# - <h2> K-Means Clustering

# In[61]:


from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch


# In[62]:


K_means_Data = music_df
K_means_Data.head()


# In[63]:


plt.figure(figsize=(30,30))
sns.heatmap(K_means_Data.corr(),annot=True)


# In[64]:


features =["energy","loudness"]
X = K_means_Data[features]
KM = KMeans(n_clusters = 6)


# In[65]:


y_pred = KM.fit_predict(X)


# In[66]:


X["Cluster_Labels"] = KM.labels_


# In[67]:


plt.figure(figsize=(30,30))
sns.scatterplot(x="energy",y="loudness", hue="Cluster_Labels", data=X)


# In[70]:


silhouette_score(X.values,KM.labels_)

