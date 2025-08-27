def get_evacueeStartNodes(itration, filename="evacuee_startNode.txt"):
    E = {}
    
    with open(filename, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):  # Process every two lines
            itr_line = lines[i].strip()
            evac_line = lines[i + 1].strip()
    
            # Extract the iteration number
            itr = int(itr_line.split('=')[1].strip())
    
            # Extract the list of evacuee IDs
            evacuee_startNode = evac_line.split('=')[1].strip()
            evac_strtnodes = eval(evacuee_startNode)  # Safely parse the list
            E[itr] = evac_strtnodes
            
    return(E[itration])

def parse_path_file(path_txt):
    with open(path_txt, "r") as f:
        lines = f.readlines()

    path_blocks, dist_blocks = {}, {}

    for i in range(len(lines)):
        if "Path:" in lines[i] and i + 1 < len(lines) and "cumDist:" in lines[i + 1]:
            try:
                # Parse the path line
                path_str = lines[i].split("Path:")[1].strip()
                path = list(map(int, path_str.split(",")))

                # Use evacuee node ID (last node in path) as key
                key = path[-1]

                # Parse cumulative distances
                dist_str = lines[i + 1].split("cumDist:")[1].strip()
                dist_str = dist_str.strip()
                if dist_str.startswith("(") and dist_str.endswith(")"):
                    dist_str = dist_str[1:-1]
                dist_list = dist_str.split("), (")
                cum = [tuple(map(float, d.replace("(", "").replace(")", "").split(","))) for d in dist_list]

                path_blocks[key] = path
                dist_blocks[key] = cum

            except Exception as e:
                print(f"Skipping block due to error: {e}")
                continue

    return path_blocks, dist_blocks
