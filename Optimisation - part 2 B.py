#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random


# In[4]:


# Read the data file
#The first row of the input file was removed manually
#The cell populated with the value 'UserId' was changed to 0 manually
file_name = "Distances.csv"
input_data = pd.read_csv(file_name, delimiter=',', header=None)
user_identifiers = input_data.iloc[0, 1:].values
dist_matrix = input_data.iloc[1:, 1:].values


# In[5]:


# Set parameters
num_centers = 6
dist_threshold = 300
duration_per_visit = 1200
max_service_duration = 8 * 60 * 60


# In[6]:


# Phase 1: Choose the best locations for the centers
def get_connectivity_and_avg_dist(matrix, node_idx):
    connected_nodes = np.nonzero(matrix[node_idx])
    connectivity = len(connected_nodes[0])
    avg_distance = np.mean(matrix[node_idx, connected_nodes])
    return connectivity, avg_distance

customers = set(range(len(user_identifiers)))
service_centers = set()

while len(service_centers) < num_centers:
    best_candidate = -1
    best_connectivity = -1
    best_avg_dist = float('inf')

    for i in customers:
        connectivity, avg_dist = get_connectivity_and_avg_dist(dist_matrix, i)
        if connectivity > best_connectivity or (connectivity == best_connectivity and avg_dist < best_avg_dist):
            best_candidate = i
            best_connectivity = connectivity
            best_avg_dist = avg_dist

    service_centers.add(best_candidate)
    customers.remove(best_candidate)

    print(f"Selected center: {best_candidate} (User ID: {user_identifiers[best_candidate]})")
    print(f"Connectivity: {best_connectivity}, Average Distance: {best_avg_dist}")
    print(f"Current centers: {service_centers}\n")

    for j in list(customers):
        if dist_matrix[best_candidate, j] < dist_threshold:
            customers.remove(j)

print("Final centers:", service_centers)


# In[ ]:


# Phase 2: Generate valid routes for each staff member
def find_optimal_starting_center(matrix, centers, remaining_clients):
    optimal_center = -1
    optimal_total_dist = float('inf')

    for center in centers:
        total_dist = np.sum(matrix[center, list(remaining_clients)])
        if total_dist < optimal_total_dist:
            optimal_center = center
            optimal_total_dist = total_dist

    return optimal_center

remaining_customers = set(range(len(user_identifiers))) - service_centers
staff_routes = []
route_number = 1

while len(remaining_customers) > 0:
    print(f"Creating Route {route_number}:")
    route = []
    service_duration = 0
    current_loc = find_optimal_starting_center(dist_matrix, service_centers, remaining_customers)
    route.append(current_loc)

    print(f"Starting center: {current_loc} (User ID: {user_identifiers[current_loc]})")

    while True:
        nearest_remaining_customer = -1
        min_dist = float('inf')

        for customer in remaining_customers:
            distance = dist_matrix[current_loc, customer]
            if distance < min_dist:
                min_dist = distance
                nearest_remaining_customer = customer

        if service_duration + min_dist + duration_per_visit > max_service_duration:
            break

        service_duration += min_dist + duration_per_visit
        route.append(nearest_remaining_customer)
        remaining_customers.remove(nearest_remaining_customer)
        current_loc = nearest_remaining_customer

        print(f"Added customer: {nearest_remaining_customer} (User ID: {user_identifiers[nearest_remaining_customer]})")

        print(f"Finished Route {route_number}: {[user_identifiers[i] for i in route]}\n")
    staff_routes.append(route)
    route_number += 1


# In[8]:


print("Centers:", [user_identifiers[i] for i in service_centers])
print("Routes:")
for i, route in enumerate(staff_routes):
    print(f"Staff {i + 1}: {[user_identifiers[i] for i in route]}")


# In[9]:


# Write the results to a text file
with open("output for 2C.txt", "w") as file:
    file.write("Centers: " + ", ".join([str(user_identifiers[i]) for i in service_centers]) + "\n")
    file.write("Routes:\n")
    for i, route in enumerate(staff_routes):
        file.write(f"Staff {i + 1}: {[user_identifiers[i] for i in route]}\n")
print("Documented results: output for 2C.txt")


# The End
