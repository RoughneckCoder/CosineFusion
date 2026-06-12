#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

py::dict cosine_similarity(py::array_t<double> A, py::array_t<double> B) {
    
    auto a = A.unchecked<2>();
    auto b = B.unchecked<2>();
    
    if (a.shape(1) != b.shape(1))
        throw std::runtime_error("Matrix mismatch"); 
    
    ssize_t n_a = a.shape(0); // Count users/rows in A -> eg. Customer
    ssize_t n_b = b.shape(0); // How many items in Row B -> eg. Items
    ssize_t dim = a.shape(1); // Count Number of columns (e.g. 128 features) -> eg. Features data points in each row
    
     //Output matrix
    py::array_t<double> result({n_a, n_b});
    auto r = result.mutable_unchecked<2>();

    // Store user lengths in norm_a and norm_b
    std::vector<double> norm_a(n_a, 0.0), norm_b(n_b, 0.0); 
    
    // compute norms for rows of A
    for (ssize_t i = 0; i < n_a; ++i) {
        double s = 0.0;
        for (ssize_t d = 0; d < dim; ++d) {
            double v = a(i, d);
            s += v * v;
        }
        norm_a[i] = std::sqrt(s);
    }
    // compute norms for rows of B
    for (ssize_t j = 0; j < n_b; ++j) {
        double s = 0.0;
        for (ssize_t d = 0; d < dim; ++d) {
            double v = b(j, d);
            s += v * v;
        }
        norm_b[j] = std::sqrt(s);
    }

    // compute cosine similarity for each pair
    for (ssize_t i = 0; i < n_a; ++i) {
        for (ssize_t j = 0; j < n_b; ++j) {
            double dot = 0.0;
            for (ssize_t d = 0; d < dim; ++d)
                dot += a(i, d) * b(j, d);
            double denom = norm_a[i] * norm_b[j];
            r(i, j) = (denom == 0.0) ? 0.0 : dot / denom;
        }
    }

    py::dict output;
    output["similarity_matrix"] = result;
    output["meta"] = py::dict(
        "n_a"_a = n_a,
        "n_b"_a = n_b,
        "dim"_a = dim,
        "metric"_a = "cosine_similarity"
    );

    return output; 
}


PYBIND11_MODULE(core_backend, m) {
    m.doc() = "C++ cosine similarity exposed to Python via pybind11 as core_backend";
    m.def("cosine_similarity", &cosine_similarity, "Compute cosine similarity",
          py::arg("A"), py::arg("B"));
}
