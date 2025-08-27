# (San Francisco Only, change the dataset for other cities)

The README is only for San Francisco dataset. To run for other dataset, replace San Francisco.csv with the new dataset.

## Quick start

- Build the C++ executable (if you change OpenMP_MOSP.cpp):
```bash
cd project && make openmp_mosp
```

- Run the San Francisco pipeline:
```bash
cd project/San_Francisco && python3 run_san_francisco_pipeline.py
```


## Flow of execution

- Step 1: Generate initial path objectives
  - Command: ../openmp_mosp -g "../CSVs/San Francisco.csv" -s 0
  - Outputs to output1/:
    - node_mapping.txt
    - San_Francisco_w1.mtx, San_Francisco_w2.mtx, San_Francisco_w3.mtx
    - San_Francisco_shortest_path_tree.txt
    - final_edge_weights_directional_path_objectives.txt
    - parents_w1.txt, parents_w2.txt, parents_w3.txt

- Step 2: Generate evacuee starting nodes
  - Command: python3 ../generate_evacuee_nodes_from_paths.py -p "output1/final_edge_weights_directional_path_objectives.txt" -e 100 -i 5
  - Produces/updates evacuee_startNode.txt and prints sample nodes
  - Pipeline then backs up/copies path objectives to backup/ and output/

- Iterations 1 → 4
  - Evacuee simulation (per iteration k)
    - Command: "../Project 2/venv/bin/python" main_create_changed_edges_modified.py -z "Original_fire_probabality_zone.csv" -w "final_edge_weights_directional.csv" -f "directional_flow_only.csv" -p "output/San_Francisco_path_objectives.txt" -e 100 -i k -o "output/iterationk"
    - Outputs: output/iterationk/updated_graph.csv, ins_edges.csv, del_edges.csv, evacuees_list.pkl
    - At k=4 also writes results_...csv

  - Update graph and regenerate paths (k < 4)
    - Update graph command:
      "../Project 2/venv/bin/python" run_update_graph.py "final_edge_weights_directional.csv" "output/iterationk/del_edges.csv" "output/iterationk/ins_edges.csv" "temp_final_edge_weights_directional.csv"
    - Regenerate path objectives command:
      ../openmp_mosp -g "temp_final_edge_weights_directional.csv" -s 0
    - Moves: output1/temp_final_edge_weights_directional_path_objectives.txt → output/San_Francisco_path_objectives.txt
    - Snapshots as: output/San_Francisco_path_objectives_iterationk.txt
    - Cleans: temp_final_edge_weights_directional.csv


## Timing information

The timing information is generated in the file timing_results_openmp.txt under San_Francisco folder.

