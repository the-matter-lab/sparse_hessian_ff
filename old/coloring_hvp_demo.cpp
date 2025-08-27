// file: coloring_hvp_demo.cpp
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "ColPackHeaders.h"
#include "utilities.h"

using namespace std;
using namespace ColPack;

// ---------- Synthetic Hessian: HVP only ----------
struct HvpEngine {
    int n;
    vector<double> a;                 // diagonal, a_i > 0
    vector<pair<int,int>> edges;      // undirected
    vector<double> w;                 // per-edge weights

    HvpEngine(int n_, vector<double> a_, vector<pair<int,int>> edges_, vector<double> w_)
        : n(n_), a(std::move(a_)), edges(std::move(edges_)), w(std::move(w_)) {}

    vector<double> apply_H(const vector<double>& v) const {
        vector<double> Hv(n, 0.0);
        for (int i = 0; i < n; ++i) Hv[i] += a[i] * v[i];
        for (size_t e = 0; e < edges.size(); ++e) {
            int u = edges[e].first, t = edges[e].second;
            double ww = w[e];
            double du = v[u] - v[t];
            Hv[u] += ww * du;
            Hv[t] += ww * (-du);
        }
        return Hv;
    }
    double Hij(int i, int j) const {
        if (i == j) {
            double val = a[i];
            for (size_t e = 0; e < edges.size(); ++e) {
                int u = edges[e].first, t = edges[e].second;
                if (u == i || t == i) val += w[e];
            }
            return val;
        } else {
            for (size_t e = 0; e < edges.size(); ++e) {
                int u = edges[e].first, t = edges[e].second;
                if ((u == i && t == j) || (u == j && t == i)) return -w[e];
            }
            return 0.0;
        }
    }
};


int main() {
    // --- Graph ---
    int n = 12;
    vector<pair<int,int>> edges = {
        {0,1},{1,2},{2,3},{3,4},{4,5},
        {5,6},{6,7},{7,8},{8,9},{9,10},{10,11},
        {0,2},{1,3},{3,7},{2,9} // chords/long hops
    };

    unsigned int** rc = make_row_compressed_from_edges(n, edges);
    string ordering = "LARGEST_FIRST";

// --- Build synthetic H and HVP engine ---
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_a(0.5, 1.5);
    std::uniform_real_distribution<double> dist_w(0.8, 1.2);
    vector<double> a(n); for (int i = 0; i < n; ++i) a[i] = dist_a(rng);
    vector<double> w(edges.size()); for (size_t e = 0; e < edges.size(); ++e) w[e] = dist_w(rng);
    HvpEngine H(n, a, edges, w);

// --- Now color with DISTANCE_TWO for direct one-hot readout ---
    GraphColoringInterface G_d2(SRC_MEM_ADOLC, rc, n);
    if (G_d2.Coloring(ordering.c_str(), "DISTANCE_TWO") != _TRUE) {
        cerr << "DISTANCE_TWO coloring failed\n";
        free_row_compressed(n, rc);
        return 2;
    }
    vector<int> d2_raw; G_d2.GetVertexColors(d2_raw);
    int K_d2 = normalize_colors_zero_based(d2_raw);

    cout << "\nDISTANCE_TWO colors (K=" << K_d2 << "):\n";
    for (int i = 0; i < n; ++i) cout << "v " << i << " -> color " << d2_raw[i] << "\n";

    

    // Build seeds and recover with naive readout
    auto seeds_d2 = build_onehot_seeds(n, K_d2, d2_raw);
    vector<vector<double>> B_d2(K_d2, vector<double>(n, 0.0));
    for (int c = 0; c < K_d2; ++c) {
        auto Hv = H.apply_H(seeds_d2[c]);
        for (int i = 0; i < n; ++i) B_d2[c][i] = Hv[i];
    }
    double max_err_d2 = 0.0;
    // diag
    for (int i = 0; i < n; ++i) {
        double rec = B_d2[d2_raw[i]][i];
        double tru = H.Hij(i, i);
        max_err_d2 = max(max_err_d2, fabs(rec - tru));
    }
    // edges
    for (auto [i, j] : edges) {
        double rec_ij = B_d2[d2_raw[j]][i];
        double tru_ij = H.Hij(i, j);
        max_err_d2 = max(max_err_d2, fabs(rec_ij - tru_ij));
    }
    cout << "First: " << H.Hij(0,0) << "\n";
    cout << "Second: " << B_d2[d2_raw[0]][0] << "\n";
    cout << "\n[DISTANCE_TWO] Naive readout max|error| (should be ~0): " << max_err_d2 << "\n";

    free_row_compressed(n, rc);
    return 0;
}



// compile with: 
// g++ -O2 -std=c++17 coloring_hvp_demo.cpp utilities.cpp -I/h/28/lucahiede/Projects/sparse_hessian_ff/ColPack/build/cmake/mywork/include/ColPack_headers -I/h/28/lucahiede/Projects/sparse_hessian_ff/ColPack/src/Utilities -I/h/28/lucahiede/Projects/sparse_hessian_ff/ColPack/src/GeneralGraphColoring -I/h/28/lucahiede/Projects/sparse_hessian_ff/ColPack/src/BipartiteGraphBicoloring -I/h/28/lucahiede/Projects/sparse_hessian_ff/ColPack/src/BipartiteGraphPartialColoring -I/h/28/lucahiede/Projects/sparse_hessian_ff/ColPack/src/Recovery -L/h/28/lucahiede/Projects/sparse_hessian_ff/ColPack/build/cmake/mywork/lib/shared_library -lColPack -ldl -o coloring_hvp_demo