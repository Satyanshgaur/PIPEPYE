#include <pipepye/sparse/cpu_ops.hpp>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#else
#include <thread>
#endif

namespace pipepye::sparse::cpu_ops {

int get_max_threads() noexcept {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    unsigned int c = std::thread::hardware_concurrency();
    return c > 0 ? static_cast<int>(c) : 1;
#endif
}

void hadamard(ConstVectorView x, ConstVectorView y, MutableVectorView out) {
    if (x.size() != y.size() || x.size() != out.size()) {
        throw std::invalid_argument("cpu_ops::hadamard dimension mismatch: x (" +
                                    std::to_string(x.size()) + "), y (" +
                                    std::to_string(y.size()) + "), out (" +
                                    std::to_string(out.size()) + ")");
    }
    for (index_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] * y[i];
    }
}

void spmv_csr(scalar_t alpha, const CSRMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y) {
    A.spmv(alpha, x, beta, y);
}

void spmv_transpose_csr(scalar_t alpha, const CSRMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y) {
    A.spmv_transpose(alpha, x, beta, y);
}

void spmv_csc(scalar_t alpha, const CSCMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y) {
    A.spmv(alpha, x, beta, y);
}

void spmv_transpose_csc(scalar_t alpha, const CSCMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y) {
    A.spmv_transpose(alpha, x, beta, y);
}

void spmv_csr_parallel(scalar_t alpha, const CSRMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y, int num_threads) {
    index_t num_rows = A.num_rows();
    index_t num_cols = A.num_cols();
    if (x.size() != num_cols) {
        throw std::invalid_argument("spmv_csr_parallel x size mismatch: expected " +
                                    std::to_string(num_cols) + ", got " + std::to_string(x.size()));
    }
    if (y.size() != num_rows) {
        throw std::invalid_argument("spmv_csr_parallel y size mismatch: expected " +
                                    std::to_string(num_rows) + ", got " + std::to_string(y.size()));
    }

    auto row_ptr = A.row_ptr();
    auto col_ind = A.col_ind();
    auto values = A.values();

#ifdef _OPENMP
    int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(guided)
#endif
    for (index_t i = 0; i < num_rows; ++i) {
        scalar_t row_dot = 0.0;
        index_t start = row_ptr[i];
        index_t end = row_ptr[i + 1];
        for (index_t k = start; k < end; ++k) {
            row_dot += values[k] * x[col_ind[k]];
        }
        if (beta == 0.0) {
            y[i] = alpha * row_dot;
        } else {
            y[i] = alpha * row_dot + beta * y[i];
        }
    }
}

void spmv_transpose_csc_parallel(scalar_t alpha, const CSCMatrix& A, ConstVectorView x, scalar_t beta, MutableVectorView y, int num_threads) {
    index_t num_rows = A.num_rows();
    index_t num_cols = A.num_cols();
    if (x.size() != num_rows) {
        throw std::invalid_argument("spmv_transpose_csc_parallel x size mismatch: expected " +
                                    std::to_string(num_rows) + ", got " + std::to_string(x.size()));
    }
    if (y.size() != num_cols) {
        throw std::invalid_argument("spmv_transpose_csc_parallel y size mismatch: expected " +
                                    std::to_string(num_cols) + ", got " + std::to_string(y.size()));
    }

    auto col_ptr = A.col_ptr();
    auto row_ind = A.row_ind();
    auto values = A.values();

#ifdef _OPENMP
    int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(guided)
#endif
    for (index_t j = 0; j < num_cols; ++j) {
        scalar_t col_dot = 0.0;
        index_t start = col_ptr[j];
        index_t end = col_ptr[j + 1];
        for (index_t k = start; k < end; ++k) {
            col_dot += values[k] * x[row_ind[k]];
        }
        if (beta == 0.0) {
            y[j] = alpha * col_dot;
        } else {
            y[j] = alpha * col_dot + beta * y[j];
        }
    }
}

void axpy_parallel(scalar_t alpha, ConstVectorView x, MutableVectorView y, int num_threads) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("axpy_parallel dimension mismatch: " +
                                    std::to_string(x.size()) + " vs " + std::to_string(y.size()));
    }
    index_t n = x.size();
#ifdef _OPENMP
    int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (index_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

scalar_t dot_parallel(ConstVectorView x, ConstVectorView y, int num_threads) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("dot_parallel dimension mismatch: " +
                                    std::to_string(x.size()) + " vs " + std::to_string(y.size()));
    }
    index_t n = x.size();
    scalar_t sum = 0.0;
#ifdef _OPENMP
    int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) reduction(+:sum) schedule(static)
#endif
    for (index_t i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

} // namespace pipepye::sparse::cpu_ops
