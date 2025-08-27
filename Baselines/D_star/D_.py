import pandas as pd
import math
from tqdm import tqdm
import contextlib
import heapq

# === File Paths ===
EDGE_FILE = "/Users/stb34/Documents/wildfire/Experiments/A*/D*/updated_graph.csv"
NODE_FILE = "/Users/stb34/Documents/wildfire/Experiments/A*/D*/Las_Vegas_node.csv"  # consider using the SF nodes file if available
EVAC_FILE = "/Users/stb34/Documents/wildfire/Experiments/A*/D*/evacuee_startNode.txt"
OUTPUT_FILE = "/Users/stb34/Documents/wildfire/Experiments/A*/D*/Las_Vegas_path_objectives.txt"
SAFE_NODE = 0   # Start node

# === Load Graph ===
edge_df = pd.read_csv(EDGE_FILE)
graph = {}
for _, row in edge_df.iterrows():
    u, v = int(row["From_Node_ID"]), int(row["To_Node_ID"])
    w1, w2, w3 = row["w1"], row["w2"], row["w3"]
    graph.setdefault(u, {})[v] = (w1, w2, w3)

# === Load Node Coordinates for Heuristic ===
node_df = pd.read_csv(NODE_FILE)
node_positions = dict(zip(node_df["Node_ID"], zip(node_df["Lon"], node_df["Lat"])))

def euclidean_heuristic(u, v):
    if u not in node_positions or v not in node_positions:
        return 0.0
    lon1, lat1 = node_positions[u]
    lon2, lat2 = node_positions[v]
    dx = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * 110540
    return math.hypot(dx, dy)

# === D* Lite Implementation ===
class DStarLite:
    def __init__(self, graph, heuristic):
        self.graph = graph
        self.h = heuristic
        self.rhs = {}
        self.g = {}
        self.U = []
        self.s_start = None
        self.s_goal = None
        self.km = 0

    def calculate_key(self, s):
        g_rhs = min(self.g.get(s, float("inf")), self.rhs.get(s, float("inf")))
        return (g_rhs + self.h(self.s_start, s) + self.km, g_rhs)

    def update_vertex(self, u):
        if u != self.s_goal:
            self.rhs[u] = min(
                [self.g.get(succ, float("inf")) + self.graph[u][succ][0]
                 for succ in self.graph.get(u, {})] or [float("inf")]
            )
        self.U = [item for item in self.U if item[1] != u]
        if self.g.get(u, float("inf")) != self.rhs.get(u, float("inf")):
            heapq.heappush(self.U, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        while (self.U and
               (self.U[0][0] < self.calculate_key(self.s_start) or
                self.rhs.get(self.s_start, float("inf")) != self.g.get(self.s_start, float("inf")))):
            k_old, u = heapq.heappop(self.U)
            k_new = self.calculate_key(u)
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
            elif self.g.get(u, float("inf")) > self.rhs.get(u, float("inf")):
                self.g[u] = self.rhs[u]
                for pred in self.graph:
                    if u in self.graph[pred]:
                        self.update_vertex(pred)
            else:
                self.g[u] = float("inf")
                self.update_vertex(u)
                for pred in self.graph:
                    if u in self.graph[pred]:
                        self.update_vertex(pred)

    def run(self, start, goal):
        self.s_start = start
        self.s_goal = goal
        self.g = {start: float("inf"), goal: float("inf")}
        self.rhs = {goal: 0}
        self.U = []
        heapq.heappush(self.U, (self.calculate_key(goal), goal))
        self.compute_shortest_path()

        # Reconstruct path
        if self.g.get(start, float("inf")) == float("inf"):
            return []
        path = [start]
        current = start
        while current != goal:
            if not self.graph.get(current):
                return []
            current = min(self.graph[current], key=lambda s: self.graph[current][s][0] + self.g.get(s, float("inf")))
            path.append(current)
        return path

# === Load Evacuee Start Nodes ===
def get_all_evacueeStartNodes(filename):
    E = set()
    with open(filename, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            evac_line = lines[i + 1].strip()
            evacuee_startNode = evac_line.split('=')[1].strip()
            E.update(eval(evacuee_startNode))
    return list(E)

# === Run from SAFE_NODE → evacuees ===
def run_dstar_from_zero():
    evacuee_nodes = get_all_evacueeStartNodes(EVAC_FILE)
    planner = DStarLite(graph, euclidean_heuristic)

    with open(OUTPUT_FILE, "w") as f, contextlib.redirect_stdout(f):
        count = 1
        for goal in tqdm(evacuee_nodes):
            path = planner.run(SAFE_NODE, goal)
            if not path:
                print(f"{count} Path: No path to {goal}")
                print(f"{count} cumDist: N/A")
                count += 1
                continue

            print(f"{count} Path: {','.join(str(n) for n in path)}")

            cum_w1 = cum_w2 = cum_w3 = 0.0
            dist_list = [(0.0, 0.0, 0.0)]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if v in graph.get(u, {}):
                    w1, w2, w3 = graph[u][v]
                    cum_w1 += w1
                    cum_w2 += w2
                    cum_w3 += w3
                    dist_list.append((cum_w1, cum_w2, cum_w3))

            dist_str = ", ".join(f"({a:.6f},{b:.6f},{c:.6f})" for a, b, c in dist_list)
            print(f"{count} cumDist: {dist_str}")
            count += 1

if __name__ == "__main__":
    run_dstar_from_zero()
    print(f"\nD* Lite from node 0 saved to: {OUTPUT_FILE}")
