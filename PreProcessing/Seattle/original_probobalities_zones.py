import pandas as pd
import networkx as nx
import random

# ✅ Ensure reproducibility
random.seed(42)

# === Step 1: Load input files ===
sf_df = pd.read_csv("/Users/stb34/Documents/wildfire/Experiments/Seattle/Seattle_link.csv")
zone3_df = pd.read_csv("/Users/stb34/Documents/wildfire/Experiments/Seattle/zone3_data.csv")

# === Step 2: Build graph ===
G = nx.Graph()
for _, row in sf_df.iterrows():
    G.add_edge(row['From_Node_ID'], row['To_Node_ID'], length=row['Length'])

# === Step 3: Extract Zone 3 nodes ===
zone3_nodes = set(zone3_df['from_node']).union(set(zone3_df['to_node']))
zone3_ros = 0.1



# === Step 5: Collect Zone 3 and non-Zone 3 nodes separately ===
zone3_results = []
nonzone3_temp = []

for node in G.nodes:
    if node in zone3_nodes:
        zone3_results.append({
            "node_id": node,
            "distance_to_zone3": 0.0,
            "ros": zone3_ros,
            "fire_prob": 1.0,
            "zone": 3
        })
    else:
        min_dist = float('inf')
        for z3 in zone3_nodes:
            try:
                path_len = nx.shortest_path_length(G, source=node, target=z3, weight='length')
                min_dist = min(min_dist, path_len)
            except nx.NetworkXNoPath:
                continue

        if min_dist == float('inf'):
            ros = None
            fire_prob = 0.0
        else:
            ros = 0.0476 * 60
            fire_prob = round(ros / min_dist, 6)

        nonzone3_temp.append({
            "node_id": node,
            "distance_to_zone3": None if min_dist == float('inf') else min_dist,
            "ros": ros,
            "fire_prob": fire_prob
        })

# === Step 6: Compute median fire_prob for non-Zone3 nodes ===
nonzone3_df = pd.DataFrame(nonzone3_temp)
median_pt = nonzone3_df['fire_prob'].median()
print(f"Median fire probability (non-Zone3 nodes): {median_pt}")

# === Step 7: Assign zones based on median threshold ===
#nonzone3_df["fire_prob"] = nonzone3_df["fire_prob"].apply(lambda p: p if p >= median_pt else 0.0)
nonzone3_df["zone"] = nonzone3_df["fire_prob"].apply(lambda p: 2 if p >= median_pt else 1)


# === Step 8: Combine and save ===
final_df = pd.concat([pd.DataFrame(zone3_results), nonzone3_df], ignore_index=True)
output_path = "/Users/stb34/Documents/wildfire/Experiments/Seattle/Original_fire_probabality_zone.csv"
final_df.to_csv(output_path, index=False)
print(f"✅ Output saved to: {output_path}")

# === Step 9: Zone count summary ===
print("📊 Zone counts:")
print(final_df['zone'].value_counts())
