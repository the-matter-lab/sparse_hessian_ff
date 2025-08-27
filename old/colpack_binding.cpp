#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ColPackHeaders.h"
#include "utilities.h"

namespace py = pybind11;

PYBIND11_MODULE(colpack_binding, m) {
    m.def("get_colpack_seeds", [](const std::vector<std::pair<int, int>>& edges, int n, const std::string& ordering, const std::string& method) {
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
        ColPack::GraphColoringInterface G_d2(SRC_MEM_ADOLC, rc, n);
        if (G_d2.Coloring(ordering.c_str(), method.c_str()) != _TRUE) {
            free_row_compressed(n, rc);
            throw std::runtime_error("Coloring failed");
        }
        std::vector<int> d2_raw;
        G_d2.GetVertexColors(d2_raw);
        int K_d2 = normalize_colors_zero_based(d2_raw);
        auto seeds_d2 = build_onehot_seeds(n, K_d2, d2_raw);
        free_row_compressed(n, rc);
        return std::make_tuple(seeds_d2, K_d2, d2_raw);
    }, "Get ColPack seeds, K_d2, and d2_raw from a sparsity pattern.", py::arg("edges"), py::arg("n"), py::arg("ordering"), py::arg("method"));
}
