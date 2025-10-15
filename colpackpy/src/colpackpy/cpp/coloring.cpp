#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ColPackHeaders.h"
#include "utilities.h"

namespace py = pybind11;

PYBIND11_MODULE(coloring, m) {
    m.def("get_seeds_from_pattern",
    [](const std::vector<std::pair<int, int>>& edges,
       int n,
       const std::string& ordering,
       std::string method) {
        // Validate ordering
        const std::set<std::string> valid_orderings = {
            "NATURAL","LARGEST_FIRST","SMALLEST_LAST","INCIDENCE_DEGREE",
            "DISTANCE_TWO_LARGEST_FIRST","DISTANCE_TWO_SMALLEST_LAST",
            "DISTANCE_TWO_INCIDENCE_DEGREE","DYNAMIC_LARGEST_FIRST","RANDOM"};
        if (!valid_orderings.count(ordering))
            throw std::invalid_argument("Invalid ordering parameter");

        if (method == "ACYCLIC") {
            method = "ACYCLIC_FOR_INDIRECT_RECOVERY";
        }
        // Validate method
        const std::set<std::string> valid_methods = {"DISTANCE_TWO", "STAR", "ACYCLIC_FOR_INDIRECT_RECOVERY"};
        if (!valid_methods.count(method))
            throw std::invalid_argument("Invalid method parameter");

        // Build ColPack graph (row-compressed from edges)
        unsigned int** rc = make_row_compressed_from_edges(n, edges);
        ColPack::GraphColoringInterface G(SRC_MEM_ADOLC, rc, n);

        // Run the requested coloring
        G.Coloring(ordering.c_str(), method.c_str());

        // === Seed matrix & colors ===
        double*** dp3_Seed = new double**;
        int *n_rows = new int, *n_cols = new int;
        (*dp3_Seed) = G.GetSeedMatrix(n_rows, n_cols);

        std::vector<std::vector<double>> seed_matrix(*n_rows, std::vector<double>(*n_cols));
        for (int i = 0; i < *n_rows; ++i)
            for (int j = 0; j < *n_cols; ++j)
                seed_matrix[i][j] = (*dp3_Seed)[i][j];

        std::vector<int> vertex_colors;
        G.GetVertexColors(vertex_colors);

        // === Optional: cached trees for ACYCLIC_FOR_INDIRECT_RECOVERY ===
        py::dict components; // {"color_pairs": [...], "edges": [...], "vertices": [...]}
        std::vector<std::pair<int,int>> comp_color_pairs;
        std::vector<std::vector<std::pair<int,int>>> comp_edges;
        std::vector<std::vector<int>> comp_vertices;

        if (method == "ACYCLIC_FOR_INDIRECT_RECOVERY") {
            // Pull ColPack’s cached structures
            std::vector<int> vi_Vertices; // CSR rowptr over vi_Edges
            std::vector<int> vi_Edges;    // CSR colind
            G.GetVertices(vi_Vertices);
            G.GetEdges(vi_Edges);

            std::map<int, std::map<int,int>> vertex_edge_id; // (u->(v->edgeID))
            G.GetVertexEdgeMap(vertex_edge_id);

            ColPack::DisjointSets ds;
            G.GetDisjointSets(ds);

            // Group edges by DSU set (each set = one 2-colored component/forest)
            std::unordered_map<int, size_t> set_to_index; // DS root -> component index

            auto add_edge_to_comp = [&](int root, int u, int v) {
                auto it = set_to_index.find(root);
                size_t idx;
                if (it == set_to_index.end()) {
                    idx = comp_edges.size();
                    set_to_index.emplace(root, idx);
                    comp_edges.emplace_back();
                    comp_vertices.emplace_back();
                    comp_color_pairs.emplace_back(-1, -1);
                } else {
                    idx = it->second;
                }
                comp_edges[idx].push_back({u, v});
                comp_vertices[idx].push_back(u);
                comp_vertices[idx].push_back(v);
                if (comp_color_pairs[idx].first == -1) {
                    int c1 = vertex_colors[u], c2 = vertex_colors[v];
                    if (c1 > c2) std::swap(c1, c2);
                    comp_color_pairs[idx] = {c1, c2};
                }
            };

            // Iterate each undirected edge once (u < v) in ColPack’s CSR
            for (int u = 0; u < n; ++u) {
                for (int p = vi_Vertices[u]; p < vi_Vertices[u+1]; ++p) {
                    int v = vi_Edges[p];
                    if (u < v) {
                        int edgeID = vertex_edge_id[u][v];
                        int root = ds.FindAndCompress(edgeID); // component id
                        add_edge_to_comp(root, u, v);
                    }
                }
            }

            // Dedup vertex lists per component
            for (auto &vs : comp_vertices) {
                std::sort(vs.begin(), vs.end());
                vs.erase(std::unique(vs.begin(), vs.end()), vs.end());
            }

            components["color_pairs"] = comp_color_pairs;
            components["edges"] = comp_edges;
            components["vertices"] = comp_vertices;
        } else {
            components["color_pairs"] = comp_color_pairs;
            components["edges"] = comp_edges;
            components["vertices"] = comp_vertices;
        }

        // Cleanup
        free_row_compressed(n, rc);
        delete dp3_Seed;
        delete n_rows;
        delete n_cols;

        // Return: (seed_matrix, vertex_colors, components)
        return py::make_tuple(seed_matrix, vertex_colors, components);
    },
    "Get a seed matrix, vertex colors, and (for acyclic) cached two-colored trees.",
    py::arg("edges"), py::arg("n"), py::arg("ordering"), py::arg("method"));
}
