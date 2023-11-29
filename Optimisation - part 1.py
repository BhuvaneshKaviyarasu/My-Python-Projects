#!/usr/bin/env python
# coding: utf-8

# PART 1: Helping Diagnosis for Cancer Patients\

# In[1]:


get_ipython().system('pip install pandas pulp')
get_ipython().system('pip install seaborn')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import pulp
import matplotlib.pyplot as plt
from scipy import stats


# In[3]:


# Task 1: Input the first 500 patients' data
data = pd.read_csv("PatientData.txt", nrows=500, header=None, usecols=range(1, 32), delimiter="\t")
diagnoses = data.iloc[:, 0].to_numpy()
attributes = data.iloc[:, 1:].to_numpy()


# In[4]:


# Task 1: Input the first 500 patients' data
data = pd.read_csv("PatientData.txt", nrows=500, header=None, usecols=range(1, 32), delimiter="\t")
diagnoses = data.iloc[:, 0].to_numpy()
attributes = data.iloc[:, 1:].to_numpy()

# Create a DataFrame for attributes and add the 'diagnosis' column
attributes_df = pd.DataFrame(attributes)
attributes_df['diagnosis'] = diagnoses

# Calculate the correlation matrix
correlation_matrix = attributes_df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("The Correlation Map")
plt.xlabel("Predicted Results and Features")
plt.ylabel("Predicted Results and Features")
plt.show()


# In[5]:


# Task 2: Using the LP model, solve for w and γ
problem = pulp.LpProblem("Cancer_Classification", pulp.LpMinimize)

w = pulp.LpVariable.dicts("w", range(1, 31), cat="Continuous")
abs_w = pulp.LpVariable.dicts("abs_w", range(1, 31), lowBound=0, cat="Continuous")
gamma = pulp.LpVariable("gamma", cat="Continuous")
error_variables = {}


# In[6]:


# Absolute value constraints for w
for i in range(1, 31):
    problem += w[i] <= abs_w[i]
    problem += -w[i] <= abs_w[i]

for idx, (diagnosis, attribute) in enumerate(zip(diagnoses, attributes)):
    error_var = pulp.LpVariable(f"error_{idx}", lowBound=0, cat="Continuous")
    error_variables[idx] = error_var

    if diagnosis == "M":
        problem += -pulp.lpDot(w.values(), attribute) + gamma + 1 <= error_var
    else:
        problem += pulp.lpDot(w.values(), attribute) - gamma + 1 <= error_var


# In[7]:


# Objective function
problem += pulp.lpSum(error_variables.values()) + pulp.lpSum(abs_w.values())

# Solve the problem
problem.solve()


# In[8]:


# Task 3: Estimate the optimal error values for each patient (only misclassified patients).
with open("output.txt", "w") as output_file:
    output_file.write("Optimal values for w:\n")
    for i in range(1, 31):
        output_file.write(f"w{i}: {w[i].value()}\n")

    output_file.write(f"\nOptimal value for gamma: {gamma.value()}\n\n")

    output_file.write("Optimal error values for misclassified patients:\n")
    for idx, error_var in error_variables.items():
        if error_var.value() > 0:
            output_file.write(f"Patient {idx + 1}: {error_var.value()}\n")


# In[9]:


# Display training set output
print("Training set output:")
print("PatientID | True Diagnosis | Predicted Diagnosis | Error")
for idx, (diagnosis, attribute, error_var) in enumerate(zip(diagnoses, attributes, error_variables.values())):
    predicted_diagnosis = "M" if pulp.lpDot(w.values(), attribute).value() - gamma.value() > 0 else "B"
    print(f"{idx + 1:9} | {diagnosis:14} | {predicted_diagnosis:18} | {error_var.value():5}")


# In[10]:


# Input file for remaining 69 patients
test_data = pd.read_csv("PatientData.txt", skiprows=range(1, 501), nrows=69, header=None, usecols=range(1, 32), delimiter="\t")
test_diagnoses = test_data.iloc[:, 0].to_numpy()
test_attributes = test_data.iloc[:, 1:].to_numpy()

optimal_w = [w[i].value() for i in range(1, 31)]
optimal_gamma = gamma.value()

# Calculate error values for remaining 69 patients
test_error_values = []

for idx, (diagnosis, attribute) in enumerate(zip(test_diagnoses, test_attributes)):
    f_x = sum([optimal_w[i] * attribute[i] for i in range(30)]) - optimal_gamma
    if diagnosis == "M" and f_x <= 0:
        error_value = 1 - f_x
    elif diagnosis == "B" and f_x > 0:
        error_value = 1 + f_x
    else:
        error_value = 0
    test_error_values.append(error_value)

# Error values for the remaining 69 patients should be in the output file
with open("output.txt", "a") as output_file:
    output_file.write("\nError values for the remaining 69 patients:\n")
    for idx, error_value in enumerate(test_error_values):
        output_file.write(f"Patient {idx + 501}: {error_value}\n")


# In[11]:


# Use the optimal weights and gamma to calculate the predicted values
predicted_values = np.array([sum([w[i+1].value() * attribute[i] for i in range(30)]) - gamma.value() for attribute in attributes])

# The attributes and predicted values are stored in a DataFrame
attributes_df = pd.DataFrame(attributes)
attributes_df['predicted_values'] = predicted_values

# Calculate summary statistics
summary_statistics = attributes_df.describe()
print("Summary statistics:")
print(summary_statistics)

# Calculate R-squared value and significance (p-value)
true_values = np.array([1 if diagnosis == "M" else -1 for diagnosis in diagnoses])
slope, intercept, r_value, p_value, _ = stats.linregress(predicted_values, true_values)
r_squared = r_value ** 2

print(f"\nR-squared value: {r_squared}")


# The End 
