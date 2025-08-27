import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt

# === Step 1: Load the San Francisco link data ===
csv_path = "/Users/stb34/Documents/wildfire/Experiments/San_Francisco_link.csv"
df = pd.read_csv(csv_path)

# === Step 2: Create an undirected graph with edge lengths ===
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row['From_Node_ID'], row['To_Node_ID'], length=row['Length'])

# Safety check
if len(G.nodes) == 0:
    raise ValueError("The graph has no nodes. Check your CSV content.")

# === Step 3: Choose a random center node as fire ignition point ===
center_node = random.choice(list(G.nodes))
print(f"Selected center node (ignition point): {center_node}")

# === Step 4: Get nodes within 2 hops of the center node ===
nearby_nodes = list(nx.single_source_shortest_path_length(G, center_node, cutoff=1).keys())
if not nearby_nodes:
    raise ValueError("No nearby nodes found. Graph might be too small or disconnected.")

# === Step 5: Randomly select initial Zone 3 seeds ===
seed_nodes = random.sample(nearby_nodes, k=min(20, len(nearby_nodes)))
print(f"Selected {len(seed_nodes)} seed nodes for Zone 3.")

# === Step 6: Expand Zone 3 to include 1-hop neighbors of each seed node ===
zone3_nodes = set(seed_nodes)
for node in seed_nodes:
    zone3_nodes.update(G.neighbors(node))

print(f"Total Zone 3 nodes after expansion: {len(zone3_nodes)}")

# === Step 7: Assign fire_prob = 1.0 to Zone 3 nodes ===
for node in zone3_nodes:
    G.nodes[node]['fire_prob'] = 1.0

# === Step 8: Build combined zone3 edge + node info ===
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

# === Step 9: Save to a single CSV file ===
zone3_df = pd.DataFrame(zone3_data)
zone3_df.to_csv("/Users/stb34/Documents/wildfire/Experiments/zone3_data.csv", index=False)
print("✅ Combined Zone 3 node+edge data saved to zone3_data.csv")

# === Step 10: Optional visualization ===
color_map = ['red' if G.nodes[n].get('fire_prob') == 1.0 else 'green' for n in G.nodes]
plt.figure(figsize=(12, 12))
nx.draw(G, node_color=color_map, node_size=10, with_labels=False)
plt.title("San Francisco Graph with Zone 3 (Unsafe) Nodes in Red")
plt.tight_layout()
plt.show()
