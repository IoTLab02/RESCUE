import pandas as pd
import numpy as np
import random

# === Step 1: Load Edge and Node Data ===
edge_file = "/Users/stb34/Documents/wildfire/Experiments/Washington/Washington_link.csv"
node_file = "/Users/stb34/Documents/wildfire/Experiments/Washington/Original_fire_Probabality_zone.csv"

sf_df = pd.read_csv(edge_file)
node_df = pd.read_csv(node_file)

# === Step 2: Shift node IDs by +1 to reserve node 0 ===
sf_df["From_Node_ID"] += 1
sf_df["To_Node_ID"] += 1
node_df["node_id"] += 1

# === Step 3: Map fire probabilities ===
fire_prob_map = dict(zip(node_df["node_id"], node_df["fire_prob"]))

# === Step 4: Compute Weights (w1 = length, w2 = fire-weighted length) ===
sf_df["w1"] = sf_df["Length"]
sf_df["p1"] = sf_df["From_Node_ID"].map(fire_prob_map)
sf_df["p2"] = sf_df["To_Node_ID"].map(fire_prob_map)
sf_df["w2"] = ((sf_df["p1"] + sf_df["p2"]) / 2) * sf_df["Length"]

# === Step 5: BPR Delay (w3_forward, w3_backward) in MINUTES ===
sf_df["Free_Speed_m_min"] = sf_df["Free_Speed"] * 1000 / 60  # convert km/h to m/min
sf_df["Capacity_veh_min"] = sf_df["Capacity"] / 60           # vehicles per minute
sf_df["t_ff"] = sf_df["Length"] / sf_df["Free_Speed_m_min"]  # free-flow travel time in minutes

np.random.seed(42)
sf_df["Flow"] = np.random.uniform(0.1, 0.9, size=len(sf_df)) * sf_df["Capacity_veh_min"]
sf_df["Flow_rev"] = np.random.uniform(0.1, 0.9, size=len(sf_df)) * sf_df["Capacity_veh_min"]

alpha, beta = 0.5, 1.8
sf_df["w3_forward"] = sf_df["t_ff"] * (1 + alpha * (sf_df["Flow"] / sf_df["Capacity_veh_min"]) ** beta)
sf_df["w3_backward"] = sf_df["t_ff"] * (1 + alpha * (sf_df["Flow_rev"] / sf_df["Capacity_veh_min"]) ** beta)

# === Step 6: Create directional weights and drop duplicates ===

# Forward edges
forward_df = sf_df[["From_Node_ID", "To_Node_ID", "w1", "w2", "w3_forward"]].copy()
forward_df.rename(columns={"w3_forward": "w3"}, inplace=True)

# Reverse edges
reverse_df = sf_df[["From_Node_ID", "To_Node_ID", "w1", "w2", "w3_backward"]].copy()
reverse_df.rename(columns={
    "From_Node_ID": "To_Node_ID",
    "To_Node_ID": "From_Node_ID",
    "w3_backward": "w3"
}, inplace=True)

# Combine and remove duplicates in (From, To)
weights_df = pd.concat([forward_df, reverse_df], ignore_index=True)
weights_df = weights_df.drop_duplicates(subset=["From_Node_ID", "To_Node_ID"])
weights_df = weights_df.sort_values(by=["From_Node_ID", "To_Node_ID"]).reset_index(drop=True)

# === Step 7: Add virtual exit edges ===
random.seed(42)
exit_nodes = random.sample(list(node_df["node_id"].unique()), 5)

virtual_exit_edges = pd.DataFrame({
    "To_Node_ID": exit_nodes,
    "From_Node_ID": [0] * 5,
    "w1": [0.0] * 5,
    "w2": [0.0] * 5,
    "w3": [0.0] * 5
})

weights_df = pd.concat([weights_df, virtual_exit_edges], ignore_index=True)
weights_df = weights_df.sort_values(by=["From_Node_ID", "To_Node_ID"]).reset_index(drop=True)

# === Step 8: Create directional flows with BPR details ===
flow_forward = sf_df[[
    "From_Node_ID", "To_Node_ID", "t_ff", "Flow", "Capacity_veh_min", "w3_forward"
]].copy()
flow_forward.rename(columns={
    "t_ff": "Tff",
    "w3_forward": "BPR_delay",
    "Capacity_veh_min": "Capacity"
}, inplace=True)

flow_backward = sf_df[[
    "From_Node_ID", "To_Node_ID", "t_ff", "Flow_rev", "Capacity_veh_min", "w3_backward"
]].copy()
flow_backward.rename(columns={
    "From_Node_ID": "To_Node_ID",
    "To_Node_ID": "From_Node_ID",
    "t_ff": "Tff",
    "Flow_rev": "Flow",
    "w3_backward": "BPR_delay",
    "Capacity_veh_min": "Capacity"
}, inplace=True)

flow_df = pd.concat([flow_forward, flow_backward], ignore_index=True)
flow_df = flow_df.drop_duplicates(subset=["From_Node_ID", "To_Node_ID"])
flow_df = flow_df.sort_values(by=["From_Node_ID", "To_Node_ID"]).reset_index(drop=True)

# === Step 9: Save both files ===
weights_path = "/Users/stb34/Documents/wildfire/Experiments/Washington/final_edge_weights_directional.csv"
flow_path = "/Users/stb34/Documents/wildfire/Experiments/Washington/directional_flow_only.csv"

weights_df.to_csv(weights_path, index=False)
flow_df.to_csv(flow_path, index=False)

print(" Edge weights (with virtual exits) saved to:", weights_path)
print("Flow-only file (with deduplicated edges) saved to:", flow_path)
print(" Exit nodes connected to virtual node 0:", exit_nodes)
