#!/usr/bin/env python
# coding: utf-8

# PART 2: Improving Operations and Strategic Decision Making

# In[3]:


get_ipython().system('pip install pulp')
import pandas as pd
import networkx as nx
import pulp


# In[4]:


#The first row of the Disctances.csv file was removed manually
#The cell with the value 'UserId' was replaced with 0 manually
def load_data(filename):
    return pd.read_csv(filename, delimiter=',', header=None, index_col=None)


# In[5]:


def create_network(df, centres, clients):
    G = nx.DiGraph()
   
    # Add zero-demand center nodes 
    G.add_nodes_from(centres, demand=0)
    # Add a client node with a demand of 1
    G.add_nodes_from(clients, demand=1)

    # Assign weights to edges between centers and clients 
    for centre_idx, centre in enumerate(centres):
        for client_idx, client in enumerate(clients):
            distance = df.iloc[centre_idx, client_idx]
            if distance != 0:
                G.add_edge(centre, client, weight=distance)
                
    # Create a dummy node with the same demand as clients
    dummy_node = max(df.index) + 1
    G.add_node(dummy_node, demand=-len(clients))

    for centre in centres:
        G.add_edge(dummy_node, centre, weight=0)

    return G


# In[6]:


def solve_minimum_cost_flow(G):
    # Minimize the total cost by using linear programming 
    prob = pulp.LpProblem("Minimum_Cost_Network_Flow", sense=pulp.LpMinimize)
    # Create integer variables
    x = pulp.LpVariable.dicts("x", G.edges(), lowBound=0, cat=pulp.LpInteger)    
    # Set the objective function 
    prob += pulp.lpSum([x[e] * G.edges[e]["weight"] for e in G.edges()])
    
    # Add constraints at each node to conserve flow 
    for v in G.nodes():
        incoming_flow = pulp.lpSum([x[(i, v)] for i in G.predecessors(v)])
        outgoing_flow = pulp.lpSum([x[(v, j)] for j in G.successors(v)])
        prob += incoming_flow - outgoing_flow == G.nodes[v]["demand"]

    # Solve the linear programming problem    
    prob.solve()
    return prob, x


# In[7]:


def display_results(prob, x, centres, G):
        # Verify an optimal solution
    if pulp.LpStatus[prob.status] == "Optimal":
        print("Total Minimum Distance:", pulp.value(prob.objective))
        
        # Initiate a cycle of iterations
        for centre in centres:
            assignments = [f"{e[1]} (distance: {G.edges[e]['weight']})"
                           for e in G.edges()
                           if e[0] == centre and x[e].value() > 0]
            print(f"{centre} -> {', '.join(assignments)}")


# In[8]:


# Print the nearest center for each client 
def print_nearest_centers(clients, x, G):
    print("\nNearest centers for each client:")
    for client in clients:
        nearest_centre = None
        min_distance = float("inf")
        for e in G.edges():
            if e[1] == client and x[e].value() > 0:
                if G.edges[e]["weight"] < min_distance:
                    min_distance = G.edges[e]["weight"]
                    nearest_centre = e[0]
        print(f"Client {client} -> Centre {nearest_centre} (distance: {min_distance})")

def main():
    filename = "Distances.csv"
    df = load_data(filename)

    centres = [71, 142, 280, 3451, 6846, 7649]
    clients = [x for x in df.index if x not in centres]

    # Print the nearest center for each client 
    G = create_network(df, centres, clients)
    # Solve the minimum cost flow problem
    prob, x = solve_minimum_cost_flow(G)
    # Display results, including total minimum distances and clients for each center 
    display_results(prob, x, centres, G)
    # Print the nearest center for each client
    print_nearest_centers(clients, x, G)

if __name__ == "__main__":
    main()


# In[9]:


def main():
    filename = "Distances.csv"
    df = load_data(filename)

    centres = [71, 142, 280, 3451, 6846, 7649]
    clients = [x for x in df.index if x not in centres]

    G = create_network(df, centres, clients)
    prob, x = solve_minimum_cost_flow(G)

    # Output the results to a text file
    output_filename = "Output for 2AB.txt"
    with open(output_filename, "w") as f:
        f.write("CentreID\tClientsAssigned\n")
        for centre in centres:
            client_count = sum([1 for e in G.edges() if e[0] == centre and x[e].value() > 0])
            f.write(f"{centre}\t{client_count}\n")
    print(f"Documented results: {output_filename}")

if __name__ == "__main__":
    main()


# The End
