#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ColPackHeaders.h"
#include "utilities.h"

namespace py = pybind11;

PYBIND11_MODULE(colpackpy, m) {
    m.def("get_colpack_seeds_simplified", [](const std::vector<std::pair<int, int>>& edges, int n, const std::string& ordering, const std::string& method) {
        // Validate ordering
        const std::set<std::string> valid_orderings = {"NATURAL", "LARGEST_FIRST", "SMALLEST_LAST", "INCIDENCE_DEGREE", "DISTANCE_TWO_LARGEST_FIRST", "DISTANCE_TWO_SMALLEST_LAST", "DISTANCE_TWO_INCIDENCE_DEGREE", "DYNAMIC_LARGEST_FIRST", "RANDOM"};
        if (valid_orderings.find(ordering) == valid_orderings.end()) {
            throw std::invalid_argument("Invalid ordering parameter");
        }

        // Validate method
        const std::set<std::string> valid_methods = {"DISTANCE_TWO", "STAR", "ACYCLIC_FOR_INDIRECT_RECOVERY"};
        if (valid_methods.find(method) == valid_methods.end()) {
            throw std::invalid_argument("Invalid method parameter");
        }

        unsigned int** rc = make_row_compressed_from_edges(n, edges);
        ColPack::GraphColoringInterface G(SRC_MEM_ADOLC, rc, n);
        G.Coloring(ordering.c_str(), method.c_str());

        double*** dp3_Seed = new double**;
        int* ip1_SeedRowCount = new int;
        int* ip1_SeedColumnCount = new int;

        (*dp3_Seed) = G.GetSeedMatrix(ip1_SeedRowCount, ip1_SeedColumnCount);

        std::vector<std::vector<double>> seed_matrix(*ip1_SeedRowCount, std::vector<double>(*ip1_SeedColumnCount));
        for (int i = 0; i < *ip1_SeedRowCount; ++i) {
            for (int j = 0; j < *ip1_SeedColumnCount; ++j) {
                seed_matrix[i][j] = (*dp3_Seed)[i][j];
            }
        }

        std::vector<int> d2_raw;
        G.GetVertexColors(d2_raw);

        free_row_compressed(n, rc);
        delete dp3_Seed;
        delete ip1_SeedRowCount;
        delete ip1_SeedColumnCount;

        return std::make_tuple(seed_matrix, d2_raw);
    }, "Get simplified ColPack seeds and d2_raw using GetSeedMatrix.", py::arg("edges"), py::arg("n"), py::arg("ordering"), py::arg("method"));
}
