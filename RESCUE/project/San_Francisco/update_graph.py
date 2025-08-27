import pandas as pd
import argparse
import os

def update_graph(original_graph_file, del_edges_file, ins_edges_file, output_file):
    """
    Update the original graph by:
    1. Removing edges from del_edges.csv
    2. Adding edges from ins_edges.csv
    """
    print(f"Updating graph: {original_graph_file}")
    print(f"  - Removing edges from: {del_edges_file}")
    print(f"  - Adding edges from: {ins_edges_file}")
    print(f"  - Output: {output_file}")
    
    # Read original graph
    original_df = pd.read_csv(original_graph_file)
    print(f"  - Original graph has {len(original_df)} edges")
    
    # Read edges to delete
    del_df = pd.read_csv(del_edges_file)
    print(f"  - Found {len(del_df)} edges to delete")
    
    # Read edges to insert
    ins_df = pd.read_csv(ins_edges_file)
    print(f"  - Found {len(ins_df)} edges to insert")
    
    # Create sets for efficient lookup
    del_edges = set()
    for _, row in del_df.iterrows():
        del_edges.add((row['From_Node_ID'], row['To_Node_ID']))
    
    # Remove edges that are in del_edges.csv
    filtered_df = original_df[~original_df.apply(
        lambda row: (row['From_Node_ID'], row['To_Node_ID']) in del_edges, axis=1
    )]
    print(f"  - After deletion: {len(filtered_df)} edges")
    
    # Add edges from ins_edges.csv
    final_df = pd.concat([filtered_df, ins_df], ignore_index=True)
    print(f"  - After insertion: {len(final_df)} edges")
    
    # Save updated graph
    final_df.to_csv(output_file, index=False)
    print(f"  - Updated graph saved to: {output_file}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update graph by applying edge changes")
    parser.add_argument("original_graph", help="Original graph CSV file")
    parser.add_argument("del_edges", help="Edges to delete CSV file")
    parser.add_argument("ins_edges", help="Edges to insert CSV file")
    parser.add_argument("output", help="Output updated graph CSV file")
    
    args = parser.parse_args()
    
    update_graph(args.original_graph, args.del_edges, args.ins_edges, args.output) 