#!/usr/bin/env python3
import random
import argparse
import os

def extract_nodes_from_path_objectives(path_objectives_file):
    """Extract all destination nodes from path objectives file"""
    nodes = []
    with open(path_objectives_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('Node_ID') and not line.startswith('='):
                if ' Path:' in line:
                    node_id = line.split(' Path:')[0].strip()
                    try:
                        nodes.append(int(node_id))
                    except ValueError:
                        continue
    return nodes

def generate_evacuee_start_nodes(nodes, num_evacuees, num_iterations):
    """Generate evacuee starting nodes from available nodes"""
    if not nodes:
        print("No nodes available for evacuee starting points")
        return
    
    print(f"Available nodes: {len(nodes)}")
    print(f"Sample nodes: {nodes[:10]}")
    
    # How many unique start nodes per iteration (5 evacuees per start node)
    num_start_nodes = max(1, num_evacuees // 5)
    
    # Generate evacuee starting nodes for each iteration
    with open("evacuee_startNode.txt", "w") as f:
        for itr in range(num_iterations):
            selected_nodes = random.sample(nodes, min(num_start_nodes, len(nodes)))
            f.write(f"itr = {itr}\n")
            f.write(f"evacuee_startNode = {selected_nodes}\n")
    
    print(f"Generated evacuee starting nodes for {num_iterations} iterations with {num_start_nodes} start nodes each")

def main():
    parser = argparse.ArgumentParser(description='Generate evacuee starting nodes from path objectives')
    parser.add_argument('-p', '--path-objectives', required=True, help='Path to path objectives file')
    parser.add_argument('-e', '--evacuees', type=int, default=25, help='Number of evacuees')
    parser.add_argument('-i', '--iterations', type=int, default=5, help='Number of iterations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path_objectives):
        print(f"Error: Path objectives file not found: {args.path_objectives}")
        return
    
    # Extract nodes from path objectives
    nodes = extract_nodes_from_path_objectives(args.path_objectives)
    
    if not nodes:
        print("Error: No valid nodes found in path objectives file")
        return
    
    # Generate evacuee starting nodes
    generate_evacuee_start_nodes(nodes, args.evacuees, args.iterations)
    print("✅ Evacuee starting nodes generated successfully")

if __name__ == "__main__":
    main() 