import argparse
import pandas as pd
import random
import os

random.seed(123)


'''
Step 1
It generates the evacuees IDs for each iteration and store them in a text file
Arguments:
    -z "Original_fire_probabality_zone.csv" -e 25 -i 5
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load input CSV files.")
    # parser.add_argument(
    #     "-g", "--graph_file",
    #     type=str,
    #     required=False,
    #     help="Path to the graph CSV file"
    # )
    parser.add_argument(
        "-z", "--zone_file",
        type=str,
        required=True,
        help="zone of nodes CSV file"
    )
    parser.add_argument(
        "-e", "--evacuees_number",
        type=int,
        required=True,
        help="evacuees_number"
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        required=True,
        help="number of iterations"
    )
    args = parser.parse_args()
    
    # # experimental variables
    evacuees_num = args.evacuees_number
    iterations = args.iterations
    
    # # Load graph files
    # graph_df = pd.read_csv(args.graph_file)
    
    # # node classification file
    node_df = pd.read_csv(args.zone_file)
 
    #  Filter eligible nodes (Zone 2 and Zone 3) 
    eligible_nodes = list(node_df[node_df['zone'].isin([2, 3])]['node_id'].astype(int))
    
    
    # If previous evacuee IDs exist, delete them
    if os.path.exists("evacuee_startNode.txt"):
        os.remove("evacuee_startNode.txt")
    
    # Create evacuee IDs and store them in a text file
    for itr in range(iterations):
        # selected_nodes = random.sample(eligible_nodes, evacuees_num)
        selected_nodes = [random.choice(eligible_nodes) for _ in range(int(evacuees_num/5))] ## Allows repetition. Imp: /5 because we are considering at least 5 evacuees are starting from the same location
        with open("evacuee_startNode.txt", "a") as f:
            f.write(f"itr = {itr}\n")
            f.write(f"evacuee_startNode = {selected_nodes}\n")
            
    