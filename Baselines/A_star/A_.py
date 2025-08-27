import pandas as pd
import networkx as nx
import math
from tqdm import tqdm
import contextlib

# === File Paths ===
EDGE_FILE = "/Users/stb34/Documents/wildfire/Experiments/San_Francisco/final_edge_weights_directional.csv"
NODE_FILE = "/Users/stb34/Documents/wildfire/Experiments/A*/Washington_node.csv"
EVAC_FILE = "/Users/stb34/Documents/wildfire/Experiments/A*/evacuee_startNode.txt"
OUTPUT_FILE = "/Users/stb34/Documents/wildfire/Experiments/A*/Washington_path_objectives.txt"
SAFE_NODE = 0

# === Load Graph ===
edge_df = pd.read_csv(EDGE_FILE)
G = nx.DiGraph()
for _, row in edge_df.iterrows():
    u = int(row["From_Node_ID"])
    v = int(row["To_Node_ID"])
    G.add_edge(u, v, w1=row["w1"], w2=row["w2"], w3=row["w3"])

# === Load Node Coordinates ===
node_df = pd.read_csv(NODE_FILE)
node_positions = dict(zip(node_df["Node_ID"], zip(node_df["Lon"], node_df["Lat"])))

# === Heuristic Function ===
def euclidean_heuristic(u, v):
    if u not in node_positions or v not in node_positions:
        return 0
    lon1, lat1 = node_positions[u]
    lon2, lat2 = node_positions[v]
    dx = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * 110540
    return math.hypot(dx, dy)

# === A* Algorithm ===
def a_star(G, start, goal, heuristic_fn):
    open_set = [(0, start)]
    came_from = {}
    g_score = {n: float('inf') for n in G.nodes()}
    g_score[start] = 0
    f_score = {n: float('inf') for n in G.nodes()}
    f_score[start] = heuristic_fn(start, goal)

    while open_set:
        _, current = open_set.pop(0)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for neighbor in G.neighbors(current):
            tentative_g = g_score[current] + G[current][neighbor]["w1"]
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic_fn(neighbor, goal)
                open_set.append((f_score[neighbor], neighbor))
                open_set.sort()
    return []

# === Load All Evacuee Nodes ===
def get_all_evacueeStartNodes(filename="evacuee_startNode.txt"):
    E = set()
    with open(filename, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            evac_line = lines[i + 1].strip()
            evacuee_startNode = evac_line.split('=')[1].strip()
            evac_strtnodes = eval(evacuee_startNode)
            E.update(evac_strtnodes)
    return list(E)

# === Run A* for Evacuees Only ===
def run_astar_for_evacuees():
    evacuee_nodes = get_all_evacueeStartNodes(EVAC_FILE)

    with open(OUTPUT_FILE, "w") as f, contextlib.redirect_stdout(f):
        count = 1
        for node in tqdm(evacuee_nodes):
            path = a_star(G, SAFE_NODE, node, euclidean_heuristic)
            if not path:
                print(f"{count} Path: No path to {node}")
                print(f"{count} cumDist: N/A")
                count += 1
                continue

            print(f"{count} Path: {','.join(str(n) for n in path)}")

            cum_w1 = cum_w2 = cum_w3 = 0.0
            dist_list = [(0.0, 0.0, 0.0)]
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                if G.has_edge(u, v):
                    edge = G[u][v]
                    cum_w1 += edge["w1"]
                    cum_w2 += edge["w2"]
                    cum_w3 += edge["w3"]
                    dist_list.append((cum_w1, cum_w2, cum_w3))

            dist_str = ", ".join(f"({a:.6f},{b:.6f},{c:.6f})" for a, b, c in dist_list)
            print(f"{count} cumDist: {dist_str}")
            count += 1

# === Run ===
if __name__ == "__main__":
    run_astar_for_evacuees()
    print(f"\n A* (evacuees only) saved to: {OUTPUT_FILE}")
