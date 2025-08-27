import argparse
import pandas as pd
import random
import os
import copy
import pickle
from collections import defaultdict
import csv
from evacuee import Evacuee
from helper import parse_path_file, get_evacueeStartNodes

random.seed(123)

'''
Modified version of main_create_changed_edges.py that accepts output directory
Arguments:
    -z "Original_fire_probabality_zone.csv" -w "final_edge_weights_directional.csv" -f "directional_flow_only.csv" -p "San Francisco_path_objectives.txt" -e 25 -i 4 -o "output/iteration1"
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load input CSV files.")
    parser.add_argument(
        "-z", "--zone_file",
        type=str,
        required=True,
        help="Path of evacuees file"
    )
    parser.add_argument(
        "-w", "--edge_weight_file",
        type=str,
        required=True,
        help="edge weights of the graph file"
    )
    parser.add_argument(
        "-f", "--trafficflow_file",
        type=str,
        required=True,
        help="trafficflow_file"
    )
    parser.add_argument(
        "-p", "--path_file",
        type=str,
        required=True,
        help="paths file name"
    )
    parser.add_argument(
        "-e", "--evacuees_number",
        type=int,
        required=True,
        help="evacuees_number"
    )
    parser.add_argument(
        "-i", "--iteration_number",
        type=int,
        required=True,
        help="iteration number should be greater than 1"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="output directory for files"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ** experimental variables **
    evacuees_num = args.evacuees_number
    itr = args.iteration_number
    itr_span = 2 # min
    current_time = itr * itr_span
    checkpoint_time = 0
    print("Running for itr: ", itr)
    
    
    # ** Read files **
    # read node classification file
    node_df = pd.read_csv(args.zone_file)
    
    # read edge weights file
    edge_df = pd.read_csv(args.edge_weight_file)
    
    # read the initial traffic flow file
    traffic_df = pd.read_csv(args.trafficflow_file)
    
    # read the MOSP paths
    path_map, cum_map = parse_path_file(args.path_file)
    
    
    
    # ** Update the Graph **
    edge_df_updated = copy.deepcopy(edge_df)
    changed_edges = set()
    
    # Estimate risk
    threshold = node_df["fire_prob"].median() # threshold for fire risk
    node_prob_map = dict(zip(node_df["node_id"], node_df["fire_prob"]))
    
    delta_t = current_time - checkpoint_time
    gamma = 0.1 # tunable parameter
    updated_node_probs = {
        node: 1 - (1 - prob) ** (gamma*delta_t)
        for node, prob in node_prob_map.items()
    }
    
    for idx, row in edge_df_updated.iterrows():
        u = row["From_Node_ID"]
        v = row["To_Node_ID"]
        w1 = row["w1"]
        w3 = row["w3"]
    
        p1 = updated_node_probs.get(u, 0.0)
        p2 = updated_node_probs.get(v, 0.0)
        p_edge = (p1 + p2) / 2
        w2_updated = p_edge * w1
    
        if p_edge > threshold:
            edge_df_updated.loc[idx, "w2"] = w2_updated
            changed_edges.add((u,v))
            
    # print("changed_edges (after risk estimation): ", len(changed_edges))    
    
    
    # Estimate traffic delay
    
    # Initialize the evacuees
    evacuees = []
    
    # load the evacuees from previous iterations
    if itr > 1:
        # Try to load from previous iteration's output directory
        prev_output_dir = args.output_dir.replace(f"iteration{itr}", f"iteration{itr-1}")
        prev_evacuees_file = os.path.join(prev_output_dir, "evacuees_list.pkl")
        if os.path.exists(prev_evacuees_file):
            with open(prev_evacuees_file, "rb") as f:
                evacuees = pickle.load(f)
    
    car_counter = len(evacuees)
    print(car_counter)
    # evacuee_start_nodes = [random.choice(list(path_map.keys())) for _ in range(int(evacuees_num/5))] ## Allows repetition. Imp: /5 because we are considering at least 5 evacuees are starting from the same location
    evacuee_start_nodes = get_evacueeStartNodes(itration = itr - 1, filename="evacuee_startNode.txt")
    for key in evacuee_start_nodes:
        if key not in path_map or key not in cum_map:
            continue
        path, cum = path_map[key], cum_map[key]
        for _ in range(5):  # we consider 5 evacuees start from the same node at a time
            evacuees.append(Evacuee(eid=f"car_{car_counter+1}", starting_time=(itr - 1)*itr_span, path=path, cum=cum))
            car_counter += 1
    
    # store the updated evacuees list
    evacuees_file = os.path.join(args.output_dir, "evacuees_list.pkl")
    with open(evacuees_file, "wb") as f:
        pickle.dump(evacuees, f)
    
    # Find the location of evacuees
    active_evacuees = [e for e in evacuees if not e.finished]
    for evac in active_evacuees:
            evac.move(current_time=current_time)
            
    edge_traffic_increment = defaultdict(int)
    edges_with_new_traffic = set()
    for evac in active_evacuees:
        
        edge = evac.current_edge()
        # print(evac.eid, ":", edge)
        if edge:
            edge_traffic_increment[edge] += 1
            edges_with_new_traffic.add(edge)
            changed_edges.add(edge)
    
    # print("changed_edges (after both risk and delay estimation): ", len(changed_edges))    
    # print(edges_with_new_traffic)
    
    alpha, beta = 0.5, 1.8
    for (u, v) in edges_with_new_traffic:
        subset = traffic_df[(traffic_df["From_Node_ID"] == u) & (traffic_df["To_Node_ID"] == v)]
        if subset.empty:
            continue
        t_ff = float(subset["Tff"].iloc[0])
        flow = float(subset["Flow"].iloc[0]) + edge_traffic_increment[(u, v)]
        capacity = float(subset["Capacity"].iloc[0])
        if capacity <= 0:
            capacity = 1.0
        w3_updated = t_ff * (1 + alpha * (flow / capacity) ** beta)
        edge_df_updated.loc[(edge_df_updated["From_Node_ID"] == u) & (edge_df_updated["To_Node_ID"] == v), "w3"] = w3_updated
        
    
    merged1 = edge_df_updated.merge(edge_df, how="outer", indicator=True)
    ins_edges = merged1[merged1["_merge"] == "right_only"].drop(columns=["_merge"])
    
    merged1 = edge_df.merge(edge_df_updated, how="outer", indicator=True)
    del_edges = merged1[merged1["_merge"] == "right_only"].drop(columns=["_merge"])
    
    
    # Save files to output directory
    updated_graph_file = os.path.join(args.output_dir, "updated_graph.csv")
    ins_edges_file = os.path.join(args.output_dir, "ins_edges.csv")
    del_edges_file = os.path.join(args.output_dir, "del_edges.csv")
    
    edge_df_updated.to_csv(updated_graph_file, index=False)
    ins_edges.to_csv(ins_edges_file, index=False)
    del_edges.to_csv(del_edges_file, index=False)
    
    if itr==4:
        results_file = os.path.join(args.output_dir, f"results_{evacuees_num}_{itr}.csv")
        with open(results_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(["eid", "total_dist_to_cover", "total_risk", "total_delay"])
            
            # Write data
            for e in evacuees:
                writer.writerow([e.eid, e.total_dist_to_cover, e.total_risk, e.total_delay]) 