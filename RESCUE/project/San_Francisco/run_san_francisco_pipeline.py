#!/usr/bin/env python3
import subprocess
import os
import shutil
import time

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f" {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f" {description} completed successfully")
            return True
        else:
            print(f" {description} failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f" {description} failed with exception: {e}")
        return False

def sanitize_timing_file(path="timing_results_openmp.txt"):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        keep = [ln for ln in lines if ("constructGraph()" not in ln and "Total Execution Time:" not in ln)]
        with open(path, "w") as f:
            f.writelines(keep)
    except Exception:
        pass

def backup_initial_paths():
    """Create backup of initial path objectives"""
    source = "output1/final_edge_weights_directional_path_objectives.txt"
    backup = "backup/San_Francisco_path_objectives.txt"
    
    if os.path.exists(source):
        print(f" Creating backup of initial path objectives")
        shutil.copy2(source, backup)
        print(f" Backup created: {backup}")
        return True
    else:
        print(f" Source file not found: {source}")
        return False

def main():
    print(" Starting San_Francisco Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Generate Initial Path Objectives (using source node 0)
    # Use parallel version
    cmd1 = f'../openmp_mosp -g "../CSVs/San Francisco.csv" -s 0'  # Adjust args to match OpenMP_MOSP
    if not run_command(cmd1, "Step 1: Generate Initial Path Objectives"):
        print("Critical failure in Step 1 - aborting city")
        return False
    
    # Step 2: Generate Evacuee Starting Nodes from path objectives (e evacuees)
    cmd2 = r'python3 ../generate_evacuee_nodes_from_paths.py -p "output1/final_edge_weights_directional_path_objectives.txt" -e 100 -i 5'
    if not run_command(cmd2, "Step 2: Generate Evacuee Starting Nodes"):
        return False
    
    # Create backup of initial paths
    if not backup_initial_paths():
        return False
    
    # Copy the generated path objectives to output folder
    if os.path.exists("output1/final_edge_weights_directional_path_objectives.txt"):
        shutil.copy2("output1/final_edge_weights_directional_path_objectives.txt", "output/San_Francisco_path_objectives.txt")
        # Also save as initial version
        shutil.copy2("output1/final_edge_weights_directional_path_objectives.txt", "output/San_Francisco_path_objectives_initial.txt")
        print(" Copied generated path objectives to output folder")
        print(" Saved initial version as San_Francisco_path_objectives_initial.txt")
    else:
        print(" Generated path objectives file not found")
        return False
    
    # Iterations 1-4
    for iteration in range(1, 5):
        print(f"\n{'='*20} ITERATION {iteration} {'='*20}")
        
        # Evacuees file is already in current directory from previous iteration
        if iteration > 1:
            print(f" Using existing evacuees_list.pkl from previous iteration")
        
        # Step: Run evacuee simulation
        cmd_sim = f'"../Project 2/venv/bin/python" main_create_changed_edges_modified.py -z "Original_fire_probabality_zone.csv" -w "final_edge_weights_directional.csv" -f "directional_flow_only.csv" -p "output/San_Francisco_path_objectives.txt" -e 100 -i {iteration} -o "output/iteration{iteration}"'
        if not run_command(cmd_sim, f"Iteration {iteration}: Evacuee Simulation"):
            return False
        
        # Files are kept in the same directory and override each iteration
        print(f" Files generated in current directory for iteration {iteration}")
        
        # Update graph and regenerate paths (except for iteration 4)
        if iteration < 4:
            print(f"\n Iteration {iteration}: Updating Graph and Regenerating Paths")
            
            # Update graph
            del_edges_file = f"output/iteration{iteration}/del_edges.csv"
            ins_edges_file = f"output/iteration{iteration}/ins_edges.csv"
            temp_graph_file = f"temp_final_edge_weights_directional.csv"
            
            if os.path.exists(del_edges_file) and os.path.exists(ins_edges_file):
                # Use virtual environment for update_graph
                cmd_update = f'"../Project 2/venv/bin/python" run_update_graph.py "final_edge_weights_directional.csv" "{del_edges_file}" "{ins_edges_file}" "{temp_graph_file}"'
                if not run_command(cmd_update, f"Iteration {iteration}: Update Graph"):
                    return False
                
                # Regenerate path objectives (using source node 0)
                cmd_paths = f'../openmp_mosp -g "{temp_graph_file}" -s 0'
                if not run_command(cmd_paths, f"Iteration {iteration}: Regenerate Path Objectives"):
                    return False
                
                # Move updated path objectives to output folder
                if os.path.exists("output1/temp_final_edge_weights_directional_path_objectives.txt"):
                    shutil.move("output1/temp_final_edge_weights_directional_path_objectives.txt", "output/San_Francisco_path_objectives.txt")
                    # Also save iteration-specific version
                    iteration_path_file = f"output/San_Francisco_path_objectives_iteration{iteration}.txt"
                    shutil.copy2("output/San_Francisco_path_objectives.txt", iteration_path_file)
                    print(" Moved updated path objectives to output folder")
                    print(f" Saved iteration {iteration} version as {iteration_path_file}")
                else:
                    print(" Updated path objectives file not found")
                    return False
                
                # Clean up temp file
                if os.path.exists(temp_graph_file):
                    os.remove(temp_graph_file)
                    print(" Cleaned up temporary graph file")
            else:
                print(f" Missing edge files for iteration {iteration}")
                return False
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(" San_Francisco Pipeline Finished Successfully!")
    print(f"  Total execution time: {total_time:.1f} seconds")
    print(f" Outputs organized in: output/")
    print(f" Backup stored in: backup/")
    print(f"{'='*60}")
    # sanitize timing file if present
    sanitize_timing_file()
    
    return True

if __name__ == "__main__":
    main()
