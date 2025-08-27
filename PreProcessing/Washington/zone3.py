import pandas as pd
import networkx as nx
import random

# === Step 1: Load the Seattle link data ===
csv_path = "/Users/stb34/Documents/wildfire/Experiments/Washington/Washington_link.csv"
df = pd.read_csv(csv_path)

# === Step 2: Create an undirected graph with edge lengths ===
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row['From_Node_ID'], row['To_Node_ID'], length=row['Length'])

if len(G.nodes) == 0:
    raise ValueError("The graph is empty!")

# === Step 3: Pick a center node and collect exactly 20 connected nodes ===
center_node = random.choice(list(G.nodes))
print(f"🎯 Center node (ignition point): {center_node}")

connected_nodes = list(nx.single_source_shortest_path_length(G, center_node).keys())
if len(connected_nodes) < 20:
    raise ValueError("Graph component too small to find 20 connected nodes.")

zone3_nodes = set(connected_nodes[:20])
print(f"✅ Selected {len(zone3_nodes)} connected Zone 3 nodes.")

# === Step 4: Assign fire_prob = 1.0 to Zone 3 nodes ===
for node in zone3_nodes:
    G.nodes[node]['fire_prob'] = 1.0

# === Step 5: Extract edges where either endpoint is in Zone 3 ===
zone3_data = []
for u, v, data in G.edges(data=True):
    if u in zone3_nodes or v in zone3_nodes:
        zone3_data.append({
            "from_node": u,
            "to_node": v,
            "length": data.get("length", None),
            "zone": 3,
            "fire_prob": 1.0
        })

# === Step 6: Save to CSV ===
zone3_df = pd.DataFrame(zone3_data)
zone3_df.to_csv("/Users/stb34/Documents/wildfire/Experiments/Washington/zone3_data.csv", index=False)
print("📁 zone3_data.csv saved with", len(zone3_df), "edges.")
