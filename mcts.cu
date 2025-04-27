#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ----------- Game State ------------
struct State {
    int board[9];    // 0 = empty, 1 = X, -1 = O
    int player;      // current player (1 = X, -1 = O)
    __host__ __device__ State() : player(1) { for (int i = 0; i < 9; i++) board[i] = 0; }
    __host__ __device__ State(const State& s) : player(s.player) { for (int i = 0; i < 9; i++) board[i] = s.board[i]; }
    __host__ __device__ void applyAction(int a) { board[a] = player; player = -player; }
    __host__ __device__ bool isTerminal() const {
        // rows and cols
        for (int i = 0; i < 3; i++) {
            int rs = board[3*i] + board[3*i+1] + board[3*i+2];
            if (abs(rs) == 3) return true;
            int cs = board[i] + board[i+3] + board[i+6];
            if (abs(cs) == 3) return true;
        }
        // diagonals
        int d1 = board[0] + board[4] + board[8];
        int d2 = board[2] + board[4] + board[6];
        if (abs(d1) == 3 || abs(d2) == 3) return true;
        // draw
        for (int i = 0; i < 9; i++) if (board[i] == 0) return false;
        return true;
    }
    __host__ __device__ int getWinner() const {
        // assumes terminal
        for (int i = 0; i < 3; i++) {
            int rs = board[3*i] + board[3*i+1] + board[3*i+2];
            if (abs(rs) == 3) return rs > 0 ? 1 : -1;
            int cs = board[i] + board[i+3] + board[i+6];
            if (abs(cs) == 3) return cs > 0 ? 1 : -1;
        }
        int d1 = board[0] + board[4] + board[8];
        if (abs(d1) == 3) return d1 > 0 ? 1 : -1;
        int d2 = board[2] + board[4] + board[6];
        if (abs(d2) == 3) return d2 > 0 ? 1 : -1;
        return 0; // draw
    }
    __host__ void getLegal(std::vector<int>& acts) const {
        acts.clear();
        for (int i = 0; i < 9; i++) if (board[i] == 0) acts.push_back(i);
    }
};

// Print board to console
void printBoard(const State& s) {
    for (int i = 0; i < 9; i++) {
        char c = (s.board[i] == 1 ? 'X' : (s.board[i] == -1 ? 'O' : '.'));
        std::cout << c;
        if (i % 3 == 2)
            std::cout << '\n';
        else
            std::cout << ' ';
    }
    std::cout << '\n';
}

// ----------- MCTS Node ------------
struct Node {
    State state;
    Node* parent;
    std::map<int, Node*> children;
    int visits;
    float value;
    Node(const State& s, Node* p = nullptr)
      : state(s), parent(p), visits(0), value(0.0f) {}
    bool isFullyExpanded() const {
        std::vector<int> legal;
        state.getLegal(legal);
        return children.size() == legal.size();
    }
    Node* expand() {
        std::vector<int> legal;
        state.getLegal(legal);
        for (int a : legal) {
            if (children.count(a) == 0) {
                State ns = state;
                ns.applyAction(a);
                Node* c = new Node(ns, this);
                children[a] = c;
                return c;
            }
        }
        return nullptr;
    }
    Node* bestChild(float cParam = 1.4f) {
        Node* best = nullptr;
        float bestScore = -1e9f;
        for (auto& kv : children) {
            Node* c = kv.second;
            float uct = (c->value / (c->visits + 1e-6f))
                        + cParam * sqrtf(logf(visits + 1) / (c->visits + 1e-6f));
            if (uct > bestScore) { bestScore = uct; best = c; }
        }
        return best;
    }
};

// ----------- CPU MCTS ------------
class MCTS_CPU {
public:
    MCTS_CPU(int iters,int rollout) : nIter(iters), nRollouts(rollout), rd(), gen(rd()) {}
    int search(const State& rootState) {
        Node* root = new Node(rootState);
        for (int i = 0; i < nIter; i++) {
            Node* node = root;
            // Selection
            while (node->isFullyExpanded() && !node->state.isTerminal())
                node = node->bestChild();
            // Expansion
            if (!node->state.isTerminal())
                node = node->expand();
            // Simulation
            auto t0 = std::chrono::high_resolution_clock::now();
            int reward = rolloutSum(node->state, nRollouts);
            auto t1 = std::chrono::high_resolution_clock::now();

            total_time += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            // num_call += 1;
            // Backpropagation
            while (node) {
                node->visits++;
                node->value += (node->state.player == rootState.player) ? reward : -reward;
                node = node->parent;
            }
        }
        std:: cout << "CPU time measurement\n";
        std:: cout << total_time / nIter;
        std:: cout << '\n';
        // Choose best action
        int bestAct = -1, maxV = -1;
        for (auto& kv : root->children) {
            if (kv.second->visits > maxV) {
                maxV = kv.second->visits;
                bestAct = kv.first;
            }
        }
        return bestAct;
    }
private:
    int nIter;
    int nRollouts;
    std::random_device rd;
    std::mt19937 gen;
    long long total_time = 0;
    int num_call = 0;

    int rolloutSum(const State& root, int N){
        int total = 0;
        for(int i=0;i<N;++i){
            State copy = root;
            total += rollout(copy);
        }

        if(total > 0){
            return 1;
        }else{
            if(total < 0){
                return -1;
            }
        }
        return 0;//draw
    }

    int rollout(State s) {
        std::uniform_int_distribution<> dist(0, 8);
        std::vector<int> acts;
        while (!s.isTerminal()) {
            s.getLegal(acts);
            s.applyAction(acts[dist(gen) % acts.size()]);
        }
        return s.getWinner();
    }
};

// ----------- GPU Rollout Kernels ------------
__global__ void setupRNG(curandState* rngStates, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &rngStates[id]);
}
__device__ bool devIsTerminal(const State& s) {
    for (int i = 0; i < 3; i++) {
        int rs = s.board[3*i] + s.board[3*i+1] + s.board[3*i+2];
        if (abs(rs) == 3) return true;
        int cs = s.board[i] + s.board[i+3] + s.board[i+6];
        if (abs(cs) == 3) return true;
    }
    int d1 = s.board[0] + s.board[4] + s.board[8];
    int d2 = s.board[2] + s.board[4] + s.board[6];
    if (abs(d1) == 3 || abs(d2) == 3) return true;
    for (int i = 0; i < 9; i++) if (s.board[i] == 0) return false;
    return true;
}
__global__ void rolloutKernel(int* d_states, int* d_results, curandState* rngStates, int nRollouts) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= nRollouts) return;
    State s;
    for (int i = 0; i < 9; i++) s.board[i] = d_states[id*10 + i];
    s.player = d_states[id*10 + 9];
    curandState localState = rngStates[id];
    while (!devIsTerminal(s)) {
        int empty[9], cnt = 0;
        for (int i = 0; i < 9; i++) if (s.board[i] == 0) empty[cnt++] = i;
        int idx = curand(&localState) % cnt;
        s.applyAction(empty[idx]);
    }
    d_results[id] = s.getWinner();
    rngStates[id] = localState;
}

class MCTS_GPU {
public:
    MCTS_GPU(int iters, int rollouts, int num_threads) : nIter(iters), nRollouts(rollouts), num_threads(num_threads) {
        cudaMalloc(&d_states, nRollouts*10*sizeof(int));
        cudaMalloc(&d_results, nRollouts*sizeof(int));
        cudaMalloc(&d_rng, nRollouts*sizeof(curandState));
        int threads = 128, blocks = (nRollouts + threads - 1) / threads;
        setupRNG<<<blocks, threads>>>(d_rng, time(NULL));
        cudaDeviceSynchronize();
    }
    ~MCTS_GPU() {
        cudaFree(d_states);
        cudaFree(d_results);
        cudaFree(d_rng);
    }
    int search(const State& rootState) {
        Node* root = new Node(rootState);
        for (int i = 0; i < nIter; i++) {
            Node* node = root;
            while (node->isFullyExpanded() && !node->state.isTerminal())
                node = node->bestChild();
            if (!node->state.isTerminal())
                node = node->expand();
            int reward = simulateBatch(node->state);
            while (node) {
                node->visits++;
                node->value += (node->state.player == rootState.player) ? reward : -reward;
                node = node->parent;
            }
        }
        std:: cout << "GPU time measurement\n";

        std:: cout << total_time / nIter;
        std:: cout << '\n';


        int bestAct = -1, maxV = -1;
        for (auto& kv : root->children) {
            if (kv.second->visits > maxV) {
                maxV = kv.second->visits;
                bestAct = kv.first;
            }
        }
        return bestAct;
    }
private:
    int nIter, nRollouts;
    int num_threads;
    long long total_time = 0;
    int *d_states, *d_results;
    curandState* d_rng;
    int simulateBatch(const State& s) {
        std::vector<int> h_states(nRollouts*10);
        for (int i = 0; i < nRollouts; i++) {
            for (int j = 0; j < 9; j++) h_states[i*10+j] = s.board[j];
            h_states[i*10+9] = s.player;
        }
        cudaMemcpy(d_states, h_states.data(), nRollouts*10*sizeof(int), cudaMemcpyHostToDevice);
        int threads = num_threads, blocks = (nRollouts + threads - 1) / threads;
        auto t0 = std::chrono::high_resolution_clock::now();

        rolloutKernel<<<blocks, threads>>>(d_states, d_results, d_rng, nRollouts);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();


        cudaDeviceSynchronize();
        std::vector<int> h_res(nRollouts);
        cudaMemcpy(h_res.data(), d_results, nRollouts*sizeof(int), cudaMemcpyDeviceToHost);
        int sum = 0;
        for (int v : h_res) sum += v;
        return (sum > nRollouts/2) ? 1 : -1;
    }
};

// ----------- Main & Game Flow ------------
int main(int argc, char** argv) {
    int num_threads = 128;
    int num_rollouts = 1000;
    if (argc > 1) {
        num_threads = std::max(1, std::min(1024, std::atoi(argv[1]))); // clamp 1..1024
        num_rollouts = std::atoi(argv[2]);
    }

    State s;
    std::cout << "Initial board:\n";
    printBoard(s);

    MCTS_CPU cpu(100, num_rollouts);
    MCTS_GPU gpu(100, num_rollouts,num_threads);
    bool useGpu = true;
    while (!s.isTerminal()) {
        int action = useGpu ? gpu.search(s) : cpu.search(s);
        s.applyAction(action);
        std::cout << (useGpu ? "GPU" : "CPU")
                  << " agent plays (" << action/3 << "," << action%3 << "):\n";
        printBoard(s);
        useGpu = !useGpu;
    }
    int w = s.getWinner();
    std::cout << "Game over. Winner: " << (w == 1 ? 'X' : w == -1 ? 'O' : 'D') << std::endl;
    return 0;
}
