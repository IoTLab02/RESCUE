#!/usr/bin/env python3
import sys
from update_graph import update_graph

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run_update_graph.py <original_graph> <del_edges> <ins_edges> <output>")
        sys.exit(1)
    
    original_graph = sys.argv[1]
    del_edges = sys.argv[2]
    ins_edges = sys.argv[3]
    output = sys.argv[4]
    
    update_graph(original_graph, del_edges, ins_edges, output)
