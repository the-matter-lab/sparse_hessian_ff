#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <utility>

unsigned int** make_row_compressed_from_edges(int n, const std::vector<std::pair<int,int>>& edges);
void free_row_compressed(int n, unsigned int** rc);
int normalize_colors_zero_based(std::vector<int>& col_raw);
std::vector<std::vector<double>> build_onehot_seeds(int n, int K, const std::vector<int>& colors);

#endif // UTILITIES_H
