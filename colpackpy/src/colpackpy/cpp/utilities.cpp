#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>

using namespace std;

unsigned int** make_row_compressed_from_edges(int n, const vector<pair<int,int>>& edges) {
    vector<vector<unsigned int>> nbrs(n);
    for (auto [u, v] : edges) {
        if (u == v) continue;
        nbrs[u].push_back((unsigned int)v);
        nbrs[v].push_back((unsigned int)u);
    }
    for (int i = 0; i < n; ++i) {
        auto& row = nbrs[i];
        sort(row.begin(), row.end());
        row.erase(unique(row.begin(), row.end()), row.end());
    }
    auto** rc = new unsigned int*[n];
    for (int i = 0; i < n; ++i) {
        int deg = (int)nbrs[i].size();
        rc[i] = new unsigned int[deg + 1];
        rc[i][0] = (unsigned int)deg;
        for (int k = 0; k < deg; ++k) rc[i][k + 1] = nbrs[i][k];
    }
    return rc;
}

void free_row_compressed(int n, unsigned int** rc) {
    for (int i = 0; i < n; ++i) delete[] rc[i];
    delete[] rc;
}

int normalize_colors_zero_based(vector<int>& col_raw) {
    int minc = *min_element(col_raw.begin(), col_raw.end());
    int maxc = *max_element(col_raw.begin(), col_raw.end());
    if (minc == 0) {
        // already 0-based
    } else if (minc >= 1) {
        for (int& c : col_raw) c -= 1;
        maxc -= 1;
    } else {
        throw runtime_error("Negative color id from ColPack.");
    }
    return maxc + 1; // K
}

vector<vector<double>> build_onehot_seeds(int n, int K, const vector<int>& colors) {
    vector<vector<double>> seeds(K, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) seeds[colors[i]][i] = 1.0;
    return seeds;
}
