#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn


# In[8]:


df = pd.read_csv("Work Dataset.csv")


# In[9]:


df.head()


# In[46]:


# df["condition"] = df[(df["Technology"] == "Wind")]

df['Wind_Power'] = df.apply(lambda x: 'Yes' if x['Technology'] == 'Wind' and x['MW'] >= 50 else 'No', axis=1)
df.head()


# In[47]:


#wind >= 50 MW

df['Wind_MW'] = np.where((df['Technology'] == 'Wind') & (df['MW'] >= 50), df['MW'], np.nan)
df.head()


# In[48]:


df['Wind'] = np.where((df['Technology'] == 'Wind') & (df['MW'] < 50), df['MW'], np.nan)
df.head()


# In[59]:


df['Wind_MW'] = df['Wind_MW'].fillna(0)
df['Wind'] = df['Wind'].fillna(0)
df.head()


# In[23]:


# dfh = pd.read_csv("Strathclyde Dataset1.csv")


# In[6]:


# dfh.head(20)


# In[ ]:


# Importing data of staff


# In[65]:


dfa = pd.read_csv("Data1.csv")
dfa.head(10)


# In[ ]:


# Linear equation


# In[ ]:


# import pulp as plp

# # Define the variables
# x1 = LpVariable("solar", lowBound=0, cat="Continuous")
# x2 = LpVariable("bess", lowBound=0, cat="Continuous")
# x3 = LpVariable("wind", lowBound=0, cat="Continuous")
# x4 = LpVariable("wind50", lowBound=0, cat="Continuous")

# # Create the problem
# # objctive = plp.LpMaximize
# objective = df["Techn"]
# problem = plp.LpProblem("LP Problem", objective)

# # Add the constraints
# problem += 85*x1 + 3*y == 13
# problem += x - y == -1

# # Define the objective function
# problem += 0

# # Solve the problem
# status = problem.solve()

# # Print the solution
# print("x =", value(x))
# print("y =", value(y))


# In[54]:


solar_count = df['Technology'].value_counts()['Solar'];
bess_count = df['Technology'].value_counts()['BESS'];
wind50_count = df['Wind_MW'].value_counts();
wind_count = df['Wind'].value_counts();
solar_count


# In[55]:


bess_count


# In[63]:


count = df.loc[(df['Technology'] == 'Wind') & (df['MW'] < 50), 'MW'].count()
count


# In[64]:


count50 = df.loc[(df['Technology'] == 'Wind') & (df['MW'] >= 50), 'MW'].count()
count50


# In[69]:


import pulp
from pulp import LpVariable


# In[71]:



# Define the variables
x1 = LpVariable("solar", lowBound=0, cat="Continuous")
x2 = LpVariable("bess", lowBound=0, cat="Continuous")
x3 = LpVariable("wind", lowBound=0, cat="Continuous")
x4 = LpVariable("wind50", lowBound=0, cat="Continuous")

# Create the problem
# objctive = plp.LpMaximize
objective = x1 + x2 + x3 + x4


# Add the constraints
constraint1 = x1 <= solar_count
constraint2 = x2 <= bess_count
constraint3 = x3 <= count
constraint4 = x4 <= count50
constraint5 = 4971*x1 + 3981*x2 + 6635*x3 + 8214*x4 <= 592207
constraint6 = 232*x1 + 147*x2 + 337*x3 + 253*x4 <= 207792
constraint7 = 1053*x1 + 421*x2 + 2527*3 + 842*x4 <= 138528
constraint8 = 1206*x1 + 0*x2 + 1551*x3 + 1034*x4 <= 138528
constraint9 = 69*x1 + 46*x2 + 345*x3 + 345*x4 <= 69264
constraint10 = 4591*x1 + 2532*x2 + 7134*x3 + 4844*x4 <= 187013
constraint11 = 1431*x1 + 550*x2 + 1431*x3 + 550*x4 <= 207792
constraint12 = 1762*x1 + 440*x2 + 1387*x3 + 286*x4 <= 110822
constraint13 = 1769*x1 + 781*x2 + 2757*x3 + 2183*x4 <= 138528

# Define the objective function
problem = pulp.LpProblem("LP Problem", pulp.LpMaximize)

problem += objective

#add constraint to problem

problem += constraint1
problem += constraint2
problem += constraint3
problem += constraint4
problem += constraint5
problem += constraint6
problem += constraint7
problem += constraint8
problem += constraint9
problem += constraint10
problem += constraint11
problem += constraint12
problem += constraint13

# Solve the problem
status = problem.solve()

# Print the solution
print("x1 =", x1.value())
print("x2 =", x2.value())
print("x3 =", x3.value())
print("x4 =", x4.value())
print("Optimal objective value=", pulp.value(problem.objective))


# In[80]:



# Define the variables
x1 = LpVariable("solar", lowBound=0, cat="Continuous")
x2 = LpVariable("bess", lowBound=0, cat="Continuous")
x3 = LpVariable("wind", lowBound=0, cat="Continuous")
x4 = LpVariable("wind50", lowBound=0, cat="Continuous")

# Create the problem
# objctive = plp.LpMaximize
objective = x1 + x2 + x3 + x4


# Add the constraints
constraint1 = x1 <= solar_count
constraint2 = x2 <= bess_count
constraint3 = x3 <= count
constraint4 = x4 <= count50
constraint5 = 4971*x1 + 3981*x2 + 6635*x3 + 8214*x4 <= 661471
constraint6 = 232*x1 + 147*x2 + 337*x3 + 253*x4 <= 2770560
constraint7 = 1053*x1 + 421*x2 + 2527*3 + 842*x4 <= 2077920
constraint8 = 1206*x1 + 0*x2 + 1551*x3 + 1034*x4 <= 2077920
constraint9 = 69*x1 + 46*x2 + 345*x3 + 345*x4 <= 1385280
constraint10 = 4591*x1 + 2532*x2 + 7134*x3 + 4844*x4 <= 1870130
constraint11 = 1431*x1 + 550*x2 + 1431*x3 + 550*x4 <= 2077920
constraint12 = 1762*x1 + 440*x2 + 1387*x3 + 286*x4 <= 1108220
constraint13 = 1769*x1 + 781*x2 + 2757*x3 + 2183*x4 <= 1385280

# Define the objective function
problem = pulp.LpProblem("LP Problem", pulp.LpMaximize)

problem += objective

#add constraint to problem

problem += constraint1
problem += constraint2
problem += constraint3
problem += constraint4
problem += constraint5
problem += constraint6
problem += constraint7
problem += constraint8
problem += constraint9
problem += constraint10
problem += constraint11
problem += constraint12
problem += constraint13

# Solve the problem
status = problem.solve()

# Print the solution
print("x1 =", x1.value())
print("x2 =", x2.value())
print("x3 =", x3.value())
print("x4 =", x4.value())
print("Optimal objective value=", pulp.value(problem.objective))


# In[81]:



# Define the variables
x1 = LpVariable("solar", lowBound=0, cat="Continuous")
x2 = LpVariable("bess", lowBound=0, cat="Continuous")
x3 = LpVariable("wind", lowBound=0, cat="Continuous")
x4 = LpVariable("wind50", lowBound=0, cat="Continuous")

# Create the problem
# objctive = plp.LpMaximize
objective = x1 + x2 + x3 + x4


# Add the constraints
constraint1 = x1 <= solar_count
constraint2 = x2 <= bess_count
constraint3 = x3 <= count
constraint4 = x4 <= count50
constraint5 = 4971*x1 + 3981*x2 + 6635*x3 + 8214*x4 <= 592207
constraint6 = 232*x1 + 147*x2 + 337*x3 + 253*x4 <= 207792
constraint7 = 1053*x1 + 421*x2 + 2527*3 + 842*x4 <= 138528
constraint8 = 1206*x1 + 0*x2 + 1551*x3 + 1034*x4 <= 138528
constraint9 = 69*x1 + 46*x2 + 345*x3 + 345*x4 <= 69264
constraint10 = 4591*x1 + 2532*x2 + 7134*x3 + 4844*x4 <= 187013
constraint11 = 1431*x1 + 550*x2 + 1431*x3 + 550*x4 <= 207792
constraint12 = 1762*x1 + 440*x2 + 1387*x3 + 286*x4 <= 110822
constraint13 = 1769*x1 + 781*x2 + 2757*x3 + 2183*x4 <= 138528

# Define the objective function
problem = pulp.LpProblem("LP Problem", pulp.LpMaximize)

problem += objective

#add constraint to problem

problem += constraint1
problem += constraint2
problem += constraint3
problem += constraint4
problem += constraint5
problem += constraint6
problem += constraint7
problem += constraint8
problem += constraint9
problem += constraint10
problem += constraint11
problem += constraint12
problem += constraint13

# Solve the problem
status = problem.solve()

# Print the solution
print("x1 =", x1.value())
print("x2 =", x2.value())
print("x3 =", x3.value())
print("x4 =", x4.value())
print("Optimal objective value=", pulp.value(problem.objective))


# In[ ]:


# 45328.45 (hour/site) / 1790474 (hours)

