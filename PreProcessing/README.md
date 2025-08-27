# Wildfire Evacuation Pre-Processing Pipeline

This repository describes the pre-processing pipeline for wildfire evacuation simulations.  
It consists of three steps: **Zone 3 Extraction**, **Fire Probability Zone Assignment**, and **Edge Weight & Flow Generation**.

---


## Pre-Processing Pipeline:  

### Step 1: Zone 3 Extraction  
· Input   
   o Road network CSV file (e.g., Seattle_link.csv)  
     available here - (https://figshare.com/articles/dataset/A_unified_and_validated_traffic_dataset_for_20_U_S_cities/24235696?file=48908890)  
   o Required columns: From_Node_ID, To_Node_ID, Length (extracted from the input graph)  

· Output  
   o zone3_data.csv containing:  
      ⁃ from_node, to_node  
      ⁃ length - distance  
      ⁃ zone - zone number  
      ⁃ fire_prob - fire probability  

Command to run:  
```
   python3 zone3.py
```

---


### Step 2: Fire Probability Zone Assignment  
· Input  
   o Road network CSV (e.g., Seattle_link.csv)  
   o Zone 3 edges CSV (e.g., zone3_data.csv) (from step 1)  

· Output  
   o Original_fire_probabality_zone.csv with columns:  
      ⁃ node_id - road id  
      ⁃ distance_to_zone3 - distance from node id to nearest zone3 node  
      ⁃ ros - Rate of Spread for each node  
      ⁃ fire_prob - fire probability  
      ⁃ zone - category of zones  

Command to run:  
```
   python3 original_Probablities_zone.py
```

---


### Step 3: Edge Weight & Flow Generation  
· Input  
   o Edge file: Seattle_link.csv  
   o Node file: Original_fire_Probabality_zone.csv (from step 2)  

· Output  
   o final_edge_weights_directional.csv - directional edges with weights (w1, w2, w3) + virtual exit node connections  
   o directional_flow_only.csv - directional flows with BPR delay details:  
        - Tff (free-flow travel time)  
        - Flow  
        - Capacity  
        - BPR_delay  

Command to run:  
```
   python3 bidirectional.py
```
