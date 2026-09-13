#include <pipepye/sparse/cpu_ops.hpp>
#include <stdexcept>
#include <string>

namespace pipepye::sparse::cpu_ops {

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

} // namespace pipepye::sparse::cpu_ops
