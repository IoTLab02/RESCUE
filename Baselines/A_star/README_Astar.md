# Running A* Evacuation Pipeline

This document explains the **A*** pipeline used for iterative wildfire evacuation simulations.  
The pipeline includes graph updates, evacuee state tracking, and final results generation across **4 iterations**.

---

```text
Step 1: Iterative Graph Update & Metrics

Files:
· A*.py → runs A* and generates evacuee paths (*_path_objectives.txt)
· main_create_changed_edges.py → updates graph edge weights (updated_graph.csv) and evacuee states across iterations

Process (per vehicle count):
1. Run A*.py
   o Input:
      - current edge weights (final_edge_weights_directional.csv for iter 1, then updated_graph.csv from previous iteration)
      - Original_fire_probabality_zone.csv
      - directional_flow_only.csv
      - evacuee_startNode.txt
      - Seattle_node.csv (or your node coord file)
        available here - (https://figshare.com/articles/dataset/A_unified_and_validated_traffic_dataset_for_20_U_S_cities/24235696?file=48908890)
   o Output:
      - *_path_objectives.txt
      
Command to run:
   python3 A*.py
```



---

```text
2. Run main_create_changed_edges.py
   o Use -i <iter> (1..4) along with zone file, edge weights, traffic flow, path file, and evacuee count.
   o Output:
      - updated_graph.csv
      - ins_edges.csv
      - del_edges.csv
      - evacuee states (evacuees_list.pkl)
   o At iteration 4:
      - also outputs results<evacuees>_4.csv (e.g., results25_4.csv)

Example 1st iteration:
python3 main_create_changed_edges.py   -z  "/Users/stb34/Documents/wildfire/Experiments/Seattle/Original_fire_probabality_zone.csv"   -w  "/Users/stb34/Documents/wildfire/Experiments/Seattle/final_edge_weights_directional.csv"   -f  "/Users/stb34/Documents/wildfire/Experiments/Seattle/directional_flow_only.csv"   -p  "/Users/stb34/Documents/wildfire/Experiments/A*/Seattle_path_objectives.txt"   -e 25   -i 1

(In the first iteration, the process uses final_edge_weights_directional.csv as the input graph and Seattle_path_objectives.txt which contains the paths generated from the first A* run.)

Example 2nd iteration:
python3 main_create_changed_edges.py   -z  "/Users/stb34/Documents/wildfire/Experiments/Seattle/Original_fire_probabality_zone.csv"   -w  "/Users/stb34/Documents/wildfire/Experiments/A*/updated_graph.csv"   -f  "/Users/stb34/Documents/wildfire/Experiments/Seattle/directional_flow_only.csv"   -p  "/Users/stb34/Documents/wildfire/Experiments/A*/Seattle_path_objectives.txt"   -e 25   -i 2

(here it uses the updated graph generated in iteration 1)
```

---

```text
3. Repeat for iterations 1 → 4
   o Always feed the new updated_graph.csv into the next A* run.
   o Stop after iteration 4; final results file is saved.

Vehicle Count Experiments:
· Repeat the entire 4-iteration cycle by changing evacuee_startNode.txt to match:
   - 50 evacuees → produces results50_4.csv
   - 75 evacuees → produces results75_4.csv
   - 100 evacuees → produces results100_4.csv
```

---

```text
Iterative Updates Note:
· During the simulation, the files ins_edges.csv, del_edges.csv, and updated_graph.csv are overwritten at every iteration.
· This means the data you see in these files at the end of the run corresponds to the last iteration executed for that city.
· If you need to keep intermediate results (e.g., from iteration 1, 2, 3), you should rename or copy those files after each run before starting the next iteration.
```
