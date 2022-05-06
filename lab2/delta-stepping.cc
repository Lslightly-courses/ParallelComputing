#include <iostream>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <map>
#include <bitset>
#include <set>
#include <algorithm>
#include <omp.h>

#ifdef DEBUG
#define DEBUGOUT(x) std::cout << x << std::endl;fflush(stdout);
#define DEBUGPRINT(...) printf(__VA_ARGS__);fflush(stdout);
#else
#define DEBUGOUT(x)
#define DEBUGPRINT(...)
#endif

#define MAXNUM 100000

using namespace std;

class Req {
public:
    bitset<MAXNUM> req_v;
    int new_cost[MAXNUM];
    Req() {
        req_v.reset();
    }
    void push(int v, int cost) {
        if (req_v[v] && new_cost[v] > cost) {
            new_cost[v] = cost;
        } else if (!req_v[v]) {
            new_cost[v] = cost;
            req_v.set(v);
        }
    }
};

typedef struct Edge{
    int w;
    int cost;
}Edge;

typedef bitset<MAXNUM> Bucket[MAXNUM];

Bucket buckets;
map<int, vector<Edge>> heavy;
map<int, vector<Edge>> light;
vector<Edge> edges[MAXNUM];
vector<int> tent;

void delta_stepping(int s, int delta, int V_NUM, int E_NUM, int n_proc);
void bellmannford(int s, int V_NUM, int E_NUM, int n_proc);
void relax(int v, int x, int delta);
bool isEmpty(Bucket &buckets);

/**
 * @brief 使用delta-stepping算法计算单源最短路径
 * 
 * @param s 源点
 * @param delta 
 * @param V_NUM 点数
 * @param E_NUM 边数
 * @param n_proc 并行线程数量
 */
void delta_stepping(int s, int delta, int V_NUM, int E_NUM, int n_proc) {
    omp_set_num_threads(n_proc);
    //  initialize node data structure
    for (int v = 0; v < V_NUM; v++) {
        for (auto edge: edges[v]) {
            auto cost = edge.cost;
            if (cost > delta) {
                if (heavy.find(v) == heavy.end())
                    heavy[v] = vector<Edge>();
                heavy[v].push_back(edge);
            } else {
                if (light.find(v) == light.end())
                    light[v] = vector<Edge>();
                light[v].push_back(edge);
            }
        }
        tent[v] = INT32_MAX;
    }
    
    //  source node at distance 0
    relax(s, 0, delta);
    int i = 0;
    
    //  some queued nodes left
    while (!isEmpty(buckets)) {
        bitset<MAXNUM> S;  //  no nodes deleted for this bucket yet
        while (buckets[i].count() != 0) {   //  New phase
            Req reqs;
            #pragma omp parallel
            {
                int id = omp_get_thread_num();
                int start = V_NUM/n_proc*id, end = V_NUM/n_proc*(id+1);
                if (id == n_proc-1) end = V_NUM;
                bitset<MAXNUM> local_set;
                local_set.reset();
                int local_cost[MAXNUM];
                set<int> updates;
                for (int v = start; v < end; v++) {
                    if (buckets[i][v]) {
                        for (auto edge: light[v]) {
                            if (local_set[edge.w] && local_cost[edge.w] > tent[v]+edge.cost) {
                                local_cost[edge.w] = tent[v]+edge.cost;
                                updates.insert(edge.w);
                            } else if (!local_set[edge.w]) {
                                local_cost[edge.w] = tent[v]+edge.cost;
                                local_set.set(edge.w);
                                updates.insert(edge.w);
                            }
                        }
                    }
                }
                #pragma omp critical
                {
                    for (auto w: updates) {
                        reqs.push(w, local_cost[w]);
                    }
                }
            }
            // cout << "get light reqs complete\n";
            S = S | buckets[i];
            buckets[i].reset(); //  remember deleted nodes
            #pragma omp parallel for
            for (int v = 0; v < V_NUM; v++) {
                if (reqs.req_v[v])
                    relax(v, reqs.new_cost[v], delta);   //  this may reinsert new nodes
            }
            // cout << "relax light complete\n";
        }
        // cout << "bucket " << i << " empty\n";
        Req reqs;
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            int start = V_NUM/n_proc*id, end = V_NUM/n_proc*(id+1);
            if (id == n_proc-1) end = V_NUM;
            bitset<MAXNUM> local_set;
            local_set.reset();
            int local_cost[MAXNUM];
            set<int> updates;
            for (int v = start; v < end; v++) {
                if (S[v]) {
                    for (auto edge: heavy[v]) {
                        if (local_set[edge.w] && local_cost[edge.w] > tent[v]+edge.cost) {
                            local_cost[edge.w] = tent[v]+edge.cost;
                            updates.insert(edge.w);
                        } else if (!local_set[edge.w]) {
                            local_cost[edge.w] = tent[v]+edge.cost;
                            local_set.set(edge.w);
                            updates.insert(edge.w);
                        }
                    }
                }
            }
            #pragma omp critical
            {
                for (auto w: updates) {
                    reqs.push(w, local_cost[w]);
                }
            }
        }
        // cout << "heavy " << i << " req complete\n";
        // cout << "heavy req complete\n";
        #pragma omp parallel for
        for (int i = 0; i < V_NUM; i++) {  //  relax previously deferred edges
            if (reqs.req_v[i])
                relax(i, reqs.new_cost[i], delta);
        }
        // cout << "heavy relax complete\n";
        i++;
    }
}

bool isEmpty(Bucket &buckets) {
    bool flag = true;
    for (auto bucket: buckets) {
        if (bucket.count() != 0) {
            return false;
        }
    }
    return true;
}

void relax(int v, int x, int delta) {
    if (x < tent[v]) {
        if (tent[v] != INT32_MAX) {
            buckets[tent[v]/delta].reset(v);
        }
        buckets[x/delta].set(v);
        tent[v] = x;
    }
}

class CostLessCmp {
public:
    bool operator()(const int &a, const int &b) const {
        if (a == b) {
            return false;
        } else if (tent[a] == tent[b]) {
            return a < b;
        } else {
            return tent[a] < tent[b];
        }
    }
};
typedef map<int, bool, CostLessCmp> CostQueue; 
void dijkstra(int s, int V_NUM, int E_NUM) {
    CostQueue tent_queue;
    for (int v = 0; v < V_NUM; v++) {
        tent[v] = INT32_MAX;
    }
    for (int v = 0; v < V_NUM; v++) {
        tent_queue[v] = true;
    }
    tent[s] = 0;
    tent_queue.erase(s);
    tent_queue.insert({s, true});
    while (!tent_queue.empty()) {
        auto iter_u = tent_queue.begin();
        tent_queue.erase(iter_u);
        auto u = iter_u->first;
        auto cost = tent[u];
        if (cost == INT32_MAX) {
            break;
        }
        DEBUGPRINT("u: %d tent: %lld\n", u, cost)
        for (auto edge: edges[u]) {
            if (tent[edge.w] == INT32_MAX) {
                tent_queue.erase(edge.w);
                tent[edge.w] = cost+edge.cost;
                tent_queue[edge.w] = true;
            } else if (cost+edge.cost < tent[edge.w]) {
                DEBUGPRINT("relax %d(cost: %lld) > %d to %lld from %lld\n", u, cost, edge.w, cost+edge.cost, tent[edge.w])
                tent_queue.erase(edge.w);
                tent[edge.w] = cost+edge.cost;
                tent_queue[edge.w] = true;
            }
        }
    }
}

size_t time_use(timeval start, timeval end) {
    return (end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec;
}
void output(string outfile, int s) {
    ofstream fout;
    fout.open(outfile);
    if (!fout.is_open()) {
        cerr << "error occurs when opening " << outfile << endl;
        exit(-1);
    }
    fout << "index\tshortest path\n";
    for (auto i = 0; i < tent.size(); i++) {
        if (tent[i] == INT32_MAX) {
            fout << i << endl;
            continue;
        }
        fout << i << "\t\t" << tent[i] << endl;
    }
    fout.close();
}

void statistic_delta_stepping(int s, int delta, int v_num, int e_num, string outfile, int n_proc) {
    timeval start, end;
    gettimeofday(&start, nullptr);
    delta_stepping(s, delta, v_num, e_num, n_proc);
    gettimeofday(&end, nullptr);
    cout << "delta stepping " << n_proc << "\t" << time_use(start, end) << " us" << endl;
    output(outfile+to_string(n_proc), s);
}

int main(int argc, char *argv[])
{
    if (argc != 7) {
        cout << "Usage ./a.out [directed graph source file] [vertice num] [edge number] [maximum cost] [source vertex] [output file]";
        return 0;
    }
    ifstream fin;
    fin.open(argv[1]);
    if (!fin.is_open()) {
        cerr << "error open " << argv[1] << endl;
        exit(-1);
    }
    int v_num = stoi(argv[2]);
    int e_num = stoi(argv[3]);
    int max_cost = stoi(argv[4]);
    int s = stoi(argv[5]);
    string outfile = argv[6];
    int delta = 75;
    DEBUGOUT("v_num " << v_num << " e_num " << e_num << " max cost " << max_cost << " source v " << s << " output " << outfile)
    tent = vector<int>(v_num, 0);

    string v_str, w_str, cost_str;
    while (fin >> v_str >> w_str >> cost_str) {
        int v = stoi(v_str), w = stoi(w_str), cost = stoi(cost_str);
        edges[v].push_back({w, cost});
    }
    fin.close();
    dijkstra(s, v_num, e_num);
    output(outfile+".ans", s);
    statistic_delta_stepping(s, delta, v_num, e_num, outfile, 1);
    statistic_delta_stepping(s, delta, v_num, e_num, outfile, 2);
    statistic_delta_stepping(s, delta, v_num, e_num, outfile, 3);
    statistic_delta_stepping(s, delta, v_num, e_num, outfile, 4);
    statistic_delta_stepping(s, delta, v_num, e_num, outfile, 5);
    statistic_delta_stepping(s, delta, v_num, e_num, outfile, 6);
    return 0;
}
