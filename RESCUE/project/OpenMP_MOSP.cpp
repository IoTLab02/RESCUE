#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <omp.h>
#include <ctime>
#include <type_traits>
#include <filesystem> // Added for filesystem::create_directories
#include <unordered_set>
#include <functional>
#include <utility>
struct pair_hash {
    template <class T1, class T2>
    size_t operator () (const std::pair<T1,T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;  // Simple hash combine
    }
};
using namespace std;
int NUM_THREADS = 4;

// Timing structure to store all timing data
struct TimingData {
    double updateShortestPath_time;
    double constructGraph_time;
    double bellmanFord_time;
    double total_time;
};

// Global timing data
TimingData globalTiming;

using Edge = pair<int, double>; // dest, weight

// Consensus graph built on compressed indices [0..n-1]
struct ConsensusGraph {
    int n = 0;
    vector<vector<pair<int,double>>> adj; // (to, weight)
};
class Tree {
public:
    virtual const std::vector<int>& getParents() const = 0;  // Pure virtual method
    virtual ~Tree() = default;  // Virtual destructor
};

class Tree1 : public Tree {
    std::vector<int> parents;
public:
    Tree1(const std::vector<int>& p) : parents(p) {}
    const std::vector<int>& getParents() const override {
        return parents;
    }
};

class Tree2 : public Tree {
    std::vector<int> parents;
public:
    Tree2(const std::vector<int>& p) : parents(p) {}
    const std::vector<int>& getParents() const override {
        return parents;
    }
};

class Tree3 : public Tree {
    std::vector<int> parents;
public:
    Tree3(const std::vector<int>& p) : parents(p) {}
    const std::vector<int>& getParents() const override {
        return parents;
    }
};

// Utilities to persist parents per objective (store original node-id parent per compressed index)
static bool writeParentsToFile(const string& path, const vector<int>& parentsOrig) {
    ofstream f(path);
    if (!f.is_open()) return false;
    for (int v : parentsOrig) f << v << "\n";
    return true;
}

static bool readParentsFromFile(const string& path, vector<int>& parentsOrig) {
    ifstream f(path);
    if (!f.is_open()) return false;
    parentsOrig.clear();
    string line;
    while (getline(f, line)) {
        if (line.empty()) { parentsOrig.push_back(0); continue; }
        parentsOrig.push_back(stoi(line));
    }
    return true;
}

static vector<vector<int>> buildChildrenFromParents(const vector<int>& parentOrig,
                                                    const unordered_map<int,int>& origToIdx,
                                                    int numNodes) {
    vector<vector<int>> children(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        int childOrig = i + 1; // idx_to_node[i] later for printing
        int parentOriginalId = parentOrig[i];
        if (parentOriginalId > 0) {
            auto it = origToIdx.find(parentOriginalId);
            if (it != origToIdx.end()) {
                int parentIdx = it->second; // 0-based
                children[parentIdx].push_back(i + 1); // store as 1-based for existing functions
            }
        }
    }
    return children;
}

static vector<double> computeDistancesFromParents(const vector<int>& parentOrig,
                                                  const vector<vector<pair<int,double>>>& graph,
                                                  const vector<int>& idx_to_node,
                                                  const unordered_map<int,int>& origToIdx,
                                                  int srcIdx) {
    int n = graph.size();
    vector<double> dist(n, numeric_limits<double>::infinity());
    dist[srcIdx] = 0.0;
    // simple repeated relaxation along parent links (tree); worst-case O(n^2), acceptable here
    bool changed = true;
    int guard = 0;
    while (changed && guard < n) {
        changed = false; guard++;
        for (int i = 0; i < n; ++i) {
            if (i == srcIdx) continue;
            int pOrig = parentOrig[i];
            if (pOrig <= 0) continue;
            auto it = origToIdx.find(pOrig);
            if (it == origToIdx.end()) continue;
            int pIdx = it->second;
            if (std::isinf(dist[pIdx])) continue;
            // find edge pIdx -> i
            double w = numeric_limits<double>::infinity();
            for (const auto& e : graph[pIdx]) {
                if (e.first == i) { w = e.second; break; }
            }
            if (!std::isinf(w) && dist[pIdx] + w < dist[i]) {
                dist[i] = dist[pIdx] + w;
                changed = true;
            }
        }
    }
    return dist;
}

static vector<vector<Edge>> buildPredecessor(const vector<vector<Edge>>& graphDT) {
    int n = graphDT.size();
    vector<vector<Edge>> pred(n);
    for (int u = 0; u < n; ++u) {
        for (const auto& e : graphDT[u]) {
            pred[e.first].emplace_back(u + 1, e.second);
        }
    }
    return pred;
}

static vector<vector<Edge>> readMTXAdj(const string& path, int& nOut) {
    ifstream f(path);
    vector<vector<Edge>> adj;
    nOut = 0;
    if (!f.is_open()) return adj;
    string line;
    if (!getline(f, line)) return adj;
    if (line.rfind("%%MatrixMarket", 0) == 0) {
        if (!getline(f, line)) return adj;
    }
    int n, m, nz;
    {
        stringstream ss(line);
        ss >> n >> m >> nz;
    }
    adj.assign(n, {});
    for (int i = 0; i < nz; ++i) {
        if (!getline(f, line)) break;
        stringstream ss(line);
        int u, v; double w; ss >> u >> v >> w;
        u--; v--;
        if (u >= 0 && u < n) adj[u].emplace_back(v, w);
    }
    nOut = n;
    return adj;
}

static void diffAdj(const vector<vector<Edge>>& prevAdj,
                    const vector<vector<Edge>>& currAdj,
                    vector<vector<Edge>>& changedEdgesDT) {
    int n = currAdj.size();
    changedEdgesDT.assign(n, {});
    vector<unordered_map<int,double>> prevMap(n), currMap(n);
    for (int u = 0; u < n; ++u) {
        for (const auto& e : prevAdj[u]) prevMap[u][e.first] = e.second;
        for (const auto& e : currAdj[u]) currMap[u][e.first] = e.second;
    }
    const double eps = 1e-12;
    for (int u = 0; u < n; ++u) {
        // deletions or weight changes
        for (const auto& kv : prevMap[u]) {
            int v = kv.first; double wPrev = kv.second;
            auto it = currMap[u].find(v);
            if (it == currMap[u].end()) {
                changedEdgesDT[u].emplace_back(v, -wPrev);
            } else {
                double wCurr = it->second;
                if (fabs(wCurr - wPrev) > eps) {
                    changedEdgesDT[u].emplace_back(v, -wPrev);
                    changedEdgesDT[u].emplace_back(v, +wCurr);
                }
            }
        }
        // insertions
        for (const auto& kv : currMap[u]) {
            int v = kv.first; double wCurr = kv.second;
            if (prevMap[u].find(v) == prevMap[u].end()) {
                changedEdgesDT[u].emplace_back(v, +wCurr);
            }
        }
    }
}

// Build consensus graph from multiple trees (parents are 1-based original IDs).
// Edge weight = 1.0 / occurrence_count to keep weights non-negative and favor consensus.
ConsensusGraph constructGraph(const std::vector<Tree*>& trees,
                              const std::vector<double>& Pref,
                              const std::unordered_map<int,int>& origToIdx,
                              int numNodes) {
    clock_t start = clock();

    // Count occurrences of each directed edge (parent -> child) across trees
    std::unordered_map<long long, int> edgeCount;
    edgeCount.reserve(static_cast<size_t>(numNodes) * 2);

    auto makeKey = [](int u, int v) -> long long {
        return (static_cast<long long>(u) << 32) ^ static_cast<long long>(v);
    };

    for (size_t index = 0; index < trees.size(); ++index) {
        const std::vector<int>& parents = trees[index]->getParents();
        // parents is 1-based original IDs at indices [0..n-1], with 0 or -1 meaning no parent
        #pragma omp parallel for
        for (int childIdx = 0; childIdx < static_cast<int>(parents.size()); ++childIdx) {
            int parentOrig = parents[childIdx];
            int childOrig = childIdx + 1;
            if (parentOrig > 0) {
                long long key = makeKey(parentOrig, childOrig);
                #pragma omp atomic
                edgeCount[key]++;
            }
        }
    }

    ConsensusGraph cg;
    cg.n = numNodes;
    cg.adj.assign(numNodes, {});

    for (const auto& kv : edgeCount) {
        // decode key
        int parentOrig = static_cast<int>(kv.first >> 32);
        int childOrig = static_cast<int>(kv.first & 0xffffffff);
        auto itU = origToIdx.find(parentOrig);
        auto itV = origToIdx.find(childOrig);
        if (itU == origToIdx.end() || itV == origToIdx.end()) continue;
        int u = itU->second; // 0-based index
        int v = itV->second;
        int count = kv.second;
        if (count <= 0) continue;
        double w = 1.0 / static_cast<double>(count); // smaller is better
        cg.adj[u].emplace_back(v, w);
    }

    clock_t end = clock();
    globalTiming.constructGraph_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    return cg;
}

// Safe Bellman-Ford over vector graph (indices 0..n-1)
bool bellmanFord(const ConsensusGraph& graph, int sourceIdx,
                 std::vector<double>& distances, std::vector<int>& newParent) {
    clock_t start = clock();

    int V = graph.n;
    distances.assign(V, std::numeric_limits<double>::infinity());
    newParent.assign(V, -1);
    distances[sourceIdx] = 0.0;

    // Collect edges list for easier iteration
    struct E { int u; int v; double w; };
    std::vector<E> edges;
    edges.reserve(V * 2);
    for (int u = 0; u < V; ++u) {
        for (const auto& p : graph.adj[u]) {
            edges.push_back({u, p.first, p.second});
        }
    }

    for (int i = 0; i < V - 1; ++i) {
        bool changed = false;
        #pragma omp parallel for reduction(||:changed)
        for (int ei = 0; ei < static_cast<int>(edges.size()); ++ei) {
            const auto& e = edges[ei];
            double du = distances[e.u];
            if (std::isinf(du)) continue;
            double nd = du + e.w;
            if (nd < distances[e.v]) {
                // atomic min update
                #pragma omp critical
                {
                    if (nd < distances[e.v]) {
                        distances[e.v] = nd;
                        newParent[e.v] = e.u;
                        changed = true;
                    }
                }
            }
        }
        if (!changed) break;
    }

    // No negative cycles expected (weights non-negative), but keep check
    for (const auto& e : edges) {
        if (!std::isinf(distances[e.u]) && distances[e.u] + e.w < distances[e.v]) {
            return false;
        }
    }

    clock_t end = clock();
    globalTiming.bellmanFord_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    return true;
}

bool operator==(const Edge& lhs, const Edge& rhs) {
    return lhs.first == rhs.first && lhs.second == rhs.second;
}

std::vector<std::vector<Edge>> convertToDT(std::ifstream& inputFile, bool isGraph, std::vector<std::vector<Edge>>& predecessor) {
    std::string line;
    int numRows, numCols, numNonZero;
    if (!std::getline(inputFile, line)) {
        std::cerr << "Error: Unable to read the header line from the input file." << std::endl;
        exit(1);
    }
    std::istringstream headerStream(line);
    if (!(headerStream >> numRows >> numCols >> numNonZero)) {
        std::cerr << "Error: Invalid header format in the input file." << std::endl;
        exit(1);
    }
    std::vector<std::vector<Edge>> DTMatrix(numRows);
    int lineCount = 1;
    while (std::getline(inputFile, line)) {
        lineCount++;
        std::istringstream lineStream(line);
        int row, col;
        double value;
        if (!(lineStream >> row >> col >> value)) {
            std::cerr << "Error: Invalid line format at line " << lineCount << " in the input file." << std::endl;
            exit(1);
        }
        if (row < 1 || row > numRows || col < 1 || col > numCols) {
            std::cerr << "Error: Invalid vertex indices at line " << lineCount << " in the input file." << std::endl;
            exit(1);
        }
        DTMatrix[row - 1].emplace_back(row - 1, value);
        if(isGraph)
        {
            predecessor.resize(numCols);
            predecessor[col - 1].emplace_back(row, value);
        } 
    }
    return DTMatrix;
}

std::vector<std::vector<int>> dijkstra(const std::vector<std::vector<Edge>>& graphDT, int sourceNode, std::vector<double>& shortestDist, std::vector<int>& parentList) {
    int numNodes = graphDT.size();
    std::vector<bool> visited(numNodes, false);
    shortestDist[sourceNode - 1] = 0;
    for (int i = 0; i < numNodes - 1; ++i) {
        int minDistNode = -1;
        double minDist = std::numeric_limits<double>::infinity();
        for (int j = 0; j < numNodes; ++j) {
            if (!visited[j] && shortestDist[j] < minDist) {
                minDist = shortestDist[j];
                minDistNode = j;
            }
        }
        if (minDistNode == -1) {
            break;
        }
        visited[minDistNode] = true;
        for (const Edge& edge : graphDT[minDistNode]) {
            int neighbor = edge.first;
            double weight = edge.second;
            if (!visited[neighbor] && shortestDist[minDistNode] + weight < shortestDist[neighbor]) {
                shortestDist[neighbor] = shortestDist[minDistNode] + weight;
            }
        }
    }
    std::vector<std::vector<int>> ssspTree(numNodes);
    std::vector<bool> cycleCheck (numNodes, false);
    for (int i = 0; i < numNodes; ++i) {
        if (shortestDist[i] != std::numeric_limits<double>::infinity()) {
            int parent = i + 1;
            for (const Edge& edge : graphDT[i]) {
                int child = edge.first + 1;
                if (shortestDist[child - 1] == shortestDist[i] + edge.second && !cycleCheck[child - 1]) {
                    ssspTree[parent - 1].push_back(child);
                    cycleCheck[child - 1 ] = true;
                    parentList[child] = parent;
                }
            }
        }
    }
    return ssspTree;
}

void printShortestPathTree(const std::vector<std::pair<int, std::vector<int>>>& parentChildSSP) {
    std::cout << "Shortest Path Tree:\n";
    for (const auto& node : parentChildSSP) {
        std::cout << "Node " << node.first << ": ";
        for (int child : node.second) {
            std::cout << child << " ";
        }
        std::cout << "\n";
    }
}


void markSubtreeAffected(std::vector<std::pair<int, std::vector<int>>>& parentChildSSP, std::vector<double>& shortestDist, std::vector<bool>& affectedNodes, std::queue<int>& affectedNodesQueue, std::vector<bool>& affectedDelNodes, int node, std::vector<int>& affectedNodesList, std::vector<int>& affectedNodesDel) {

    #pragma omp parallel for
    for (size_t i = 0; i < parentChildSSP[node].second.size(); ++i) {
        int child = parentChildSSP[node].second[i];
        if( !affectedDelNodes[child - 1])
        {
            affectedDelNodes[child - 1] = true;
            affectedNodesList[child - 1] = 1; 
            affectedNodesDel[child - 1] = 1;
        }
    }
}

std::vector<std::pair<int, std::vector<int>>> convertToParentChildSSP(const std::vector<std::vector<int>>& ssspTree) {
    std::vector<std::pair<int, std::vector<int>>> parentChildSSP(ssspTree.size());
    for (int i = 0; i < ssspTree.size(); ++i) {
        parentChildSSP[i].first = i + 1;  
        parentChildSSP[i].second = ssspTree[i]; 
    }
    return parentChildSSP;
}


/*
1. ssspTree is a subset (0-indexed) of graphDT having the edges belongs to shortest path. However, ssspTree contains a set of pairs with parent (1-indexed) and children (1-indexed).
2. graphDT is the whole graph containg a set of vector of edges. It is 0-indexed. i index contains all the outgoing edges (also 0-indexed) from vertex (i+1). The graph is assumed to have vertex id > 0. 
3. shortestDist is the set of all vertices (0-indexed) from the source node assumed to be 1 (in 1-index).
4. parentList is the set of all parents (1-indexed) given the child node (1-indexed) of the ssspTree.
5. Global variable Predecessor is similar to graphDT, however, instead of storing into rows, it stores in column index to find all incident vertices. Again the column index is 0-indexed, but edges are 1-indexed.
*/

void updateShortestPath(std::vector<std::pair<int, std::vector<int>>>& ssspTree,
                        std::vector<std::vector<Edge>>& graphDT,
                        const std::vector<std::vector<Edge>>& changedEdgesDT,
                        std::vector<double>& shortestDist,
                        std::vector<int>& parentList,
                        std::vector<std::vector<Edge>>& predecessor) {
    const int n = graphDT.size();
    vector<int> affected(n, 0);

    // Apply deletions and mark affected nodes if their parent edge was removed
    for (int u = 0; u < n; ++u) {
        for (const auto& e : changedEdgesDT[u]) {
            int v = e.first; double w = e.second;
            if (w < 0) {
                // remove edge u->v from graph
                auto& out = graphDT[u];
                for (auto it = out.begin(); it != out.end(); ++it) {
                    if (it->first == v) { out.erase(it); break; }
                }
                // remove from predecessor
                auto& preds = predecessor[v];
                for (auto it = preds.begin(); it != preds.end(); ++it) {
                    if (it->first == u + 1) { preds.erase(it); break; }
                }
                // if parentList[v+1] == u+1 then break tree and mark affected subtree
                if (parentList[v + 1] == u + 1) {
                    shortestDist[v] = numeric_limits<double>::infinity();
                    parentList[v + 1] = -1;
                    affected[v] = 1;
                    // mark children of v affected
                    for (int child : ssspTree[v].second) {
                        affected[child - 1] = 1;
                    }
                }
            }
        }
    }

    // Apply insertions: relax distances and update parents
    for (int u = 0; u < n; ++u) {
        for (const auto& e : changedEdgesDT[u]) {
            int v = e.first; double w = e.second;
            if (w > 0) {
                graphDT[u].push_back({v, w});
                predecessor[v].emplace_back(u + 1, w);
                if (!std::isinf(shortestDist[u]) && shortestDist[u] + w < shortestDist[v]) {
                    shortestDist[v] = shortestDist[u] + w;
                    parentList[v + 1] = u + 1;
                    affected[v] = 1;
                }
            }
        }
    }

    // Propagate improvements from affected set (parallel frontier relaxations)
    vector<int> frontier;
    for (int i = 0; i < n; ++i) if (affected[i]) frontier.push_back(i);
    while (!frontier.empty()) {
        vector<int> next;
        #pragma omp parallel
        {
            vector<int> local;
            #pragma omp for nowait
            for (int idx = 0; idx < (int)frontier.size(); ++idx) {
                int u = frontier[idx];
                for (const auto& e : graphDT[u]) {
                    int v = e.first; double w = e.second;
                    double du = shortestDist[u];
                    if (std::isinf(du)) continue;
                    double nd = du + w;
                    if (nd < shortestDist[v]) {
                        #pragma omp critical
                        {
                            if (nd < shortestDist[v]) {
                                shortestDist[v] = nd;
                                parentList[v + 1] = u + 1;
                                local.push_back(v);
                            }
                        }
                    }
                }
            }
            #pragma omp critical
            next.insert(next.end(), local.begin(), local.end());
        }
        sort(next.begin(), next.end());
        next.erase(unique(next.begin(), next.end()), next.end());
        frontier.swap(next);
    }
}
    
int find_key(const map<int,int>& m, int val) {
    for (auto& p : m) if (p.second == val) return p.first;
    return -1;
}

int main(int argc, char** argv) {
    // Parse arguments
    string graphFile, changedEdgesFile = "";
    int sourceNode = 1;
    for (int i = 1; i < argc; i += 2) {
        string option(argv[i]);
        string argument(argv[i+1]);
        if (option == "-g") graphFile = argument;
        else if (option == "-c") changedEdgesFile = argument;
        else if (option == "-s") sourceNode = stoi(argument);
    }
    string csv_filename = graphFile;
    // Strip quotes if present
    if (csv_filename.front() == '\"' && csv_filename.back() == '\"') csv_filename = csv_filename.substr(1, csv_filename.size() - 2);
    size_t last_slash = csv_filename.find_last_of('/');
    string city_name = (last_slash != string::npos) ? csv_filename.substr(last_slash + 1) : csv_filename;
    size_t last_dot = city_name.find_last_of('.');
    if (last_dot != string::npos) city_name = city_name.substr(0, last_dot);
    cout << "Step 1: Converting " << city_name << " CSV to internal format..." << endl;
    // Parse CSV
    ifstream csv(csv_filename);
    if (!csv.is_open()) { cerr << "Error: Could not open " << csv_filename << endl; return 1; }
    string line;
    getline(csv, line); // Skip header
    set<int> nodes;
    map<pair<int,int>, tuple<double,double,double>> edge_data;
    vector<vector<pair<int,double>>> graphAvg, graphW1, graphW2, graphW3;
    map<int, int> node_to_idx;
    int max_node = 0;
    while (getline(csv, line)) {
        stringstream ss(line);
        string token;
        int from, to;
        double w1, w2, w3;
        getline(ss, token, ','); from = stoi(token);
        getline(ss, token, ','); to = stoi(token);
        getline(ss, token, ','); w1 = stod(token);
        getline(ss, token, ','); w2 = stod(token);
        getline(ss, token, ','); w3 = stod(token);
        edge_data[{from, to}] = {w1, w2, w3};
        nodes.insert(from);
        nodes.insert(to);
        max_node = max({max_node, from, to});
    }
    csv.close();
    int num_nodes = nodes.size();
    int idx = 0;
    for (int node : nodes) node_to_idx[node] = idx++;
    // Build reverse mapping: internal index -> original node id
    vector<int> idx_to_node(num_nodes);
    for (const auto& kv : node_to_idx) {
        idx_to_node[kv.second] = kv.first;
    }
    graphAvg.resize(num_nodes);
    graphW1.resize(num_nodes);
    graphW2.resize(num_nodes);
    graphW3.resize(num_nodes);
    for (auto& ed : edge_data) {
        int u = node_to_idx[ed.first.first];
        int v = node_to_idx[ed.first.second];
        double w1, w2, w3;
        tie(w1, w2, w3) = ed.second;
        double avg = (w1 + w2 + w3) / 3.0;
        graphAvg[u].emplace_back(v, avg);
        graphW1[u].emplace_back(v, w1);
        graphW2[u].emplace_back(v, w2);
        graphW3[u].emplace_back(v, w3);
    }
    // Read previous MTX before overwriting to enable diff
    string w1_mtx_path = "output1/" + city_name + "_w1.mtx";
    string w2_mtx_path = "output1/" + city_name + "_w2.mtx";
    string w3_mtx_path = "output1/" + city_name + "_w3.mtx";
    vector<vector<Edge>> prevAdjW1, prevAdjW2, prevAdjW3;
    int tmpN;
    if (std::filesystem::exists(w1_mtx_path)) prevAdjW1 = readMTXAdj(w1_mtx_path, tmpN);
    if (std::filesystem::exists(w2_mtx_path)) prevAdjW2 = readMTXAdj(w2_mtx_path, tmpN);
    if (std::filesystem::exists(w3_mtx_path)) prevAdjW3 = readMTXAdj(w3_mtx_path, tmpN);

    // Helper: full Dijkstra computation returning parents as original IDs
    auto computeParents = [&](const vector<vector<pair<int,double>>>& g, int sIdx, vector<int>& parentsOut) {
        vector<double> dist(num_nodes, numeric_limits<double>::infinity());
        vector<int> parent(num_nodes, -1);
        dist[sIdx] = 0;
        priority_queue<pair<double,int>, vector<pair<double,int>>, greater<pair<double,int>>> pq;
        pq.emplace(0.0, sIdx);
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            for (const auto& e : g[u]) {
                int v = e.first; double w = e.second;
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    parent[v] = u;
                    pq.emplace(dist[v], v);
                }
            }
        }
        parentsOut.assign(num_nodes, 0);
        for (int i = 0; i < num_nodes; ++i) {
            parentsOut[i] = (parent[i] == -1 ? 0 : idx_to_node[parent[i]]); // store original ID of parent (1-based IDs in idx_to_node)
        }
    };

    int src = node_to_idx[sourceNode];
    vector<int> parentW1(num_nodes, 0), parentW2(num_nodes, 0), parentW3(num_nodes, 0);

    // Try to load previous parents; if not exist, compute fresh
    vector<int> prevParentsW1, prevParentsW2, prevParentsW3;
    bool havePrevParents = readParentsFromFile("output1/" + city_name + "_parents_w1.txt", prevParentsW1)
                        && readParentsFromFile("output1/" + city_name + "_parents_w2.txt", prevParentsW2)
                        && readParentsFromFile("output1/" + city_name + "_parents_w3.txt", prevParentsW3)
                        && prevParentsW1.size() == (size_t)num_nodes
                        && prevParentsW2.size() == (size_t)num_nodes
                        && prevParentsW3.size() == (size_t)num_nodes
                        && !prevAdjW1.empty() && !prevAdjW2.empty() && !prevAdjW3.empty();

    if (!havePrevParents) {
        // First run or missing state: compute in parallel from scratch
        #pragma omp parallel sections
        {
            #pragma omp section
            { computeParents(graphW1, src, parentW1); }
            #pragma omp section
            { computeParents(graphW2, src, parentW2); }
            #pragma omp section
            { computeParents(graphW3, src, parentW3); }
        }
    } else {
        // Incremental update using changed edges and updateShortestPath
        unordered_map<int,int> origToIdx; for (const auto& kv : node_to_idx) origToIdx[kv.first] = kv.second;

        // Prepare per-objective changed edges
        vector<vector<Edge>> changedW1, changedW2, changedW3;
        diffAdj(prevAdjW1, graphW1, changedW1);
        diffAdj(prevAdjW2, graphW2, changedW2);
        diffAdj(prevAdjW3, graphW3, changedW3);

        // Build predecessors from current graphs
        vector<vector<Edge>> predW1 = buildPredecessor(graphW1);
        vector<vector<Edge>> predW2 = buildPredecessor(graphW2);
        vector<vector<Edge>> predW3 = buildPredecessor(graphW3);

        // Build initial distances from previous parents
        vector<double> distW1 = computeDistancesFromParents(prevParentsW1, graphW1, idx_to_node, origToIdx, src);
        vector<double> distW2 = computeDistancesFromParents(prevParentsW2, graphW2, idx_to_node, origToIdx, src);
        vector<double> distW3 = computeDistancesFromParents(prevParentsW3, graphW3, idx_to_node, origToIdx, src);

        // Build ssspTree and parentList (compressed indices+1) from previous parents
        auto buildSSSP = [&](const vector<int>& parentOrig) {
            vector<pair<int, vector<int>>> sssp(num_nodes);
            for (int i = 0; i < num_nodes; ++i) { sssp[i].first = i + 1; }
            vector<vector<int>> children = buildChildrenFromParents(parentOrig, origToIdx, num_nodes);
            for (int i = 0; i < num_nodes; ++i) sssp[i].second = children[i];
            vector<int> parentList(num_nodes + 1, -1);
            for (int i = 0; i < num_nodes; ++i) {
                int pOrig = parentOrig[i];
                if (pOrig > 0) {
                    auto it = origToIdx.find(pOrig);
                    if (it != origToIdx.end()) parentList[i + 1] = it->second + 1;
                }
            }
            return make_pair(sssp, parentList);
        };

        auto res1 = buildSSSP(prevParentsW1);
        auto res2 = buildSSSP(prevParentsW2);
        auto res3 = buildSSSP(prevParentsW3);
        auto sssp1 = res1.first;      auto parentList1 = res1.second;
        auto sssp2 = res2.first;      auto parentList2 = res2.second;
        auto sssp3 = res3.first;      auto parentList3 = res3.second;

        // Update in parallel
        #pragma omp parallel sections
        {
            #pragma omp section
            { updateShortestPath(sssp1, graphW1, changedW1, distW1, parentList1, predW1); }
            #pragma omp section
            { updateShortestPath(sssp2, graphW2, changedW2, distW2, parentList2, predW2); }
            #pragma omp section
            { updateShortestPath(sssp3, graphW3, changedW3, distW3, parentList3, predW3); }
        }

        // Convert back to original-id parent arrays
        auto toParentOrig = [&](const vector<int>& parentList)->vector<int>{
            vector<int> out(num_nodes, 0);
            for (int i = 0; i < num_nodes; ++i) {
                int pIdx1 = parentList[i + 1];
                if (pIdx1 > 0) out[i] = idx_to_node[pIdx1 - 1]; else out[i] = 0;
            }
            return out;
        };
        parentW1 = toParentOrig(parentList1);
        parentW2 = toParentOrig(parentList2);
        parentW3 = toParentOrig(parentList3);
    }

    // Persist parents for next iteration
    writeParentsToFile("output1/" + city_name + "_parents_w1.txt", parentW1);
    writeParentsToFile("output1/" + city_name + "_parents_w2.txt", parentW2);
    writeParentsToFile("output1/" + city_name + "_parents_w3.txt", parentW3);

    // Wrap trees
    Tree1 t1(parentW1); Tree2 t2(parentW2); Tree3 t3(parentW3);
    vector<Tree*> trees = { &t1, &t2, &t3 };
    vector<double> Pref = {1.0, 1.0, 1.0};

    // Build consensus graph over compressed indices and run Bellman-Ford
    unordered_map<int,int> origToIdx;
    for (const auto& kv : node_to_idx) origToIdx[kv.first] = kv.second;
    ConsensusGraph cg = constructGraph(trees, Pref, origToIdx, num_nodes);
    vector<double> bfDist; vector<int> bfParentIdx;
    bool ok = bellmanFord(cg, src, bfDist, bfParentIdx);
    if (!ok) {
        cerr << "Warning: consensus graph reported negative cycle; results may be invalid" << endl;
    }
    // Save node_mapping.txt
    std::filesystem::create_directories("output1");
    ofstream mapping("output1/node_mapping.txt");
    for (int i = 0; i < num_nodes; ++i) mapping << idx_to_node[i] << " " << (i + 1) << endl;
    // Save w1.mtx, w2.mtx, w3.mtx
    ofstream w1_out("output1/" + city_name + "_w1.mtx");
    w1_out << "%%MatrixMarket matrix coordinate real general\n" << num_nodes << " " << num_nodes << " " << edge_data.size() << endl;
    ofstream w2_out("output1/" + city_name + "_w2.mtx");
    w2_out << "%%MatrixMarket matrix coordinate real general\n" << num_nodes << " " << num_nodes << " " << edge_data.size() << endl;
    ofstream w3_out("output1/" + city_name + "_w3.mtx");
    w3_out << "%%MatrixMarket matrix coordinate real general\n" << num_nodes << " " << num_nodes << " " << edge_data.size() << endl;
    for (auto& ed : edge_data) {
        int u = node_to_idx[ed.first.first] + 1;
        int v = node_to_idx[ed.first.second] + 1;
        double w1, w2, w3;
        tie(w1, w2, w3) = ed.second;
        w1_out << u << " " << v << " " << fixed << setprecision(6) << w1 << endl;
        w2_out << u << " " << v << " " << fixed << setprecision(6) << w2 << endl;
        w3_out << u << " " << v << " " << fixed << setprecision(6) << w3 << endl;
    }
    // Save consensus shortest_path_tree.txt (NodeOrig: ParentOrig)
    ofstream tree("output1/" + city_name + "_shortest_path_tree.txt");
    for (int i = 0; i < num_nodes; ++i) {
        int node_orig = idx_to_node[i];
        int p_orig = (bfParentIdx[i] == -1 ? 0 : idx_to_node[bfParentIdx[i]]);
        tree << node_orig << ": " << p_orig << endl;
    }
    // Save path_objectives.txt
    string obj_path = (city_name == "temp_final_edge_weights_directional")
        ? string("output1/temp_final_edge_weights_directional_path_objectives.txt")
        : string("output1/final_edge_weights_directional_path_objectives.txt");
    ofstream obj(obj_path);
    for (int dest = 0; dest < num_nodes; ++dest) {
        if (dest == src) continue;
        // Reconstruct path from consensus parents
        vector<int> pathIdx;
        for (int at = dest; at != -1; at = bfParentIdx[at]) pathIdx.push_back(at);
        reverse(pathIdx.begin(), pathIdx.end());
        if (pathIdx.empty() || pathIdx.front() != src) continue;
        int dest_orig = idx_to_node[dest];
        obj << dest_orig << " Path: ";
        for (int p : pathIdx) obj << idx_to_node[p] << " ";
        obj << endl;
        obj << dest_orig << " cumDist: ";
        double c1 = 0, c2 = 0, c3 = 0;
        for (size_t i = 0; i + 1 < pathIdx.size(); ++i) {
            int u_orig = idx_to_node[pathIdx[i]];
            int v_orig = idx_to_node[pathIdx[i+1]];
            auto it = edge_data.find({u_orig, v_orig});
            if (it != edge_data.end()) {
                double w1, w2, w3;
                tie(w1, w2, w3) = it->second;
                c1 += w1; c2 += w2; c3 += w3;
                obj << "(" << fixed << setprecision(6) << c1 << "," << c2 << "," << c3 << ")";
            }
        }
        obj << endl;
    }
    cout << "Step 1 completed." << endl;
    return 0;
}

