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
    import re
    with open(path_txt, "r") as f:
        lines = f.readlines()

    path_blocks, dist_blocks = {}, {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "Path:" in line:
            try:
                # Extract key before 'Path:'
                key_str = line.split("Path:")[0].strip()
                key = int(key_str.split()[0])
            except Exception:
                i += 1
                continue

            # Extract all integers in path line (handles spaces or commas)
            path_str = line.split("Path:", 1)[1].strip()
            path = [int(x) for x in re.findall(r"\d+", path_str)]

            # Next line expected to be cumDist
            cum = []
            if i + 1 < len(lines) and "cumDist:" in lines[i + 1]:
                dist_str = lines[i + 1].split("cumDist:", 1)[1].strip()
                # Find all '(a,b,c)' groups and parse floats
                tuple_groups = re.findall(r"\(([^)]*)\)", dist_str)
                for grp in tuple_groups:
                    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", grp)]
                    if len(nums) >= 3:
                        cum.append((nums[0], nums[1], nums[2]))

            if path and cum:
                path_blocks[key] = path
                dist_blocks[key] = cum

            i += 2
        else:
            i += 1

    return path_blocks, dist_blocks