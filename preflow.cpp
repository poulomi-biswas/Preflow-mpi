#include <mpi.h>
#include <iostream>
#include <fstream>
#include <climits>
#include <vector>
#include <queue>
using namespace std;

struct Edge {
    int to, capacity, flow;
};

int preflow_push(vector<vector<Edge>>& graph, int n, int source, int sink);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n, m;
    int source, sink;

    if (rank == 0) {
        ifstream inputFile("input.txt");  
        if (!inputFile.is_open()) {
            cerr << "Failed to open input file!" << endl;
            MPI_Finalize();
            return 1;
        }

        n = 2;  
        m = 0;  

        while (!inputFile.eof()) {
            string line;
            getline(inputFile, line);
            if (!line.empty()) {
                m++;  
            }
        }

        vector<vector<Edge>> graph(n);
        inputFile.clear();
        inputFile.seekg(0, ios::beg);  

        source = 0;  
        sink = 1;    

        for (int i = 0; i < m; i++) {
            int capacity;
            inputFile >> capacity;
            graph[0].push_back({1, capacity, 0});  
            graph[1].push_back({0, 0, 0});          
        }

        inputFile.close();  

        int flow = preflow_push(graph, n, source, sink);
        int flow_gather;

        MPI_Reduce(&flow, &flow_gather, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "\nMax Flow: " << flow_gather << endl;
        }
    } else {
      
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&sink, 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<vector<Edge>> graph(n);
        for (int i = 0; i < m; i++) {
            int capacity;
            MPI_Bcast(&capacity, 1, MPI_INT, 0, MPI_COMM_WORLD);
            graph[0].push_back({1, capacity, 0});  
            graph[1].push_back({0, 0, 0});          
        }

        preflow_push(graph, n, source, sink);
    }

    MPI_Finalize();

    return 0;
}

int preflow_push(vector<vector<Edge>>& graph, int n, int source, int sink) {
    vector<int> height(n, 0);
    vector<int> excess(n, 0);
    vector<int> count(n * 2, 0);

    height[source] = n;
    excess[source] = INT_MAX;

    for (Edge& edge : graph[source]) {
        int v = edge.to;
        int capacity = edge.capacity;
        graph[v].push_back({source, 0, 0});  
        excess[v] += capacity;
        edge.flow = capacity;
    }

    queue<int> active_nodes;
    for (int i = 0; i < n; i++) {
        if (i != source && i != sink && excess[i] > 0) {
            active_nodes.push(i);
        }
    }

    while (!active_nodes.empty()) {
        int u = active_nodes.front();
        active_nodes.pop();

        for (Edge& edge : graph[u]) {
            int v = edge.to;
            if (height[u] > height[v] && edge.flow < edge.capacity) {
                int amount = min(excess[u], edge.capacity - edge.flow);
                edge.flow += amount;
                graph[v][count[v]].flow -= amount;
                excess[u] -= amount;
                excess[v] += amount;

                if (v != source && v != sink && excess[v] > 0) {
                    active_nodes.push(v);
                }
            }
            count[u]++;
        }

        if (excess[u] > 0 && u != source && u != sink) {
            int old_height = height[u];
            height[u] = n * 2;

            for (const Edge& edge : graph[u]) {
                if (edge.flow < edge.capacity) {
                    height[u] = min(height[u], height[edge.to] + 1);
                }
            }

            if (height[u] > old_height) {
                active_nodes.push(u);
            }
        }
    }

    return excess[sink];
}
