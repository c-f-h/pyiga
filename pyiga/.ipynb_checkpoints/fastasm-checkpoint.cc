/*
 * fastasm.cc
 *
 * Black box fast low-rank assembling routines for Isogeometric Analysis.
 *
 * Clemens Hofreither, 2017.
 * Free for academic use.
 */
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>
#include <assert.h>
#include <cstdlib>
#include <algorithm>

//#define USE_CBLAS

#ifdef USE_CBLAS
extern "C" {
#include <cblas.h>
}
#endif


////////////////////////////////////////////////////////////////////////////////
// Logging
////////////////////////////////////////////////////////////////////////////////

typedef void (*LogFuncPtr)(const char * str, size_t bytes);

class FunctionStringBuf : public std::stringbuf
{
public:
    FunctionStringBuf(LogFuncPtr funcptr_)
        : funcptr(funcptr_)
    { }
    void set_func(LogFuncPtr funcptr_) {
        funcptr = funcptr_;
    }
    virtual int sync() {
        funcptr(this->str().c_str(), this->str().size());
        this->str("");
        return 0;
    }
private:
    LogFuncPtr funcptr;
};

void cout_logfunc(const char * str, size_t)
{
    std::cout << str;
}

FunctionStringBuf log_stringbuf(cout_logfunc);
std::ostream logger(&log_stringbuf);

extern void set_log_func(LogFuncPtr logfunc)
{
    if (logfunc)
        log_stringbuf.set_func(logfunc);
    else
        log_stringbuf.set_func(cout_logfunc);
}


////////////////////////////////////////////////////////////////////////////////
// Array classes
////////////////////////////////////////////////////////////////////////////////

template <typename T>
class Array2D
{
public:
    Array2D(size_t rows_, size_t cols_)
        : m(rows_), n(cols_), data(rows_ * cols_)
    {
    }

    size_t rows() const    { return m; }
    size_t cols() const    { return n; }

          T& operator()(size_t i, size_t j)       { return data[i*n + j]; }
    const T& operator()(size_t i, size_t j) const { return data[i*n + j]; }

    void fill(T value)
    {
        std::fill(data.begin(), data.end(), value);
    }

    void fill(const T * values)
    {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                (*this)(i,j) = *values++;
    }

    void add_cross(const std::vector<T>& x, const std::vector<T>& y)
    {
        assert(x.size() == m);
        assert(y.size() == n);
#ifdef USE_CBLAS
        /* It's possible to use BLAS to compute the rank 1 update, but in
         * my tests the speedups were insignificant.
         */
        cblas_dger(CblasRowMajor, m, n, 1.0, &x[0], 1, &y[0], 1, &data[0], n);
#else
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                (*this)(i, j) += x[i] * y[j];
#endif
    }

    void argmax_abs(size_t& out_i, size_t& out_j) const
    {
        assert((m > 0) && (n > 0));

        out_i = out_j = 0;
        double max = std::abs((*this)(0,0));

        for (size_t i = 0; i < rows(); ++i)
            for (size_t j = 0; j < cols(); ++j)
            {
                double val = std::abs((*this)(i, j));
                if (val > max)
                {
                    max = val;
                    out_i = i;
                    out_j = j;
                }
            }
    }

    void swap(Array2D<T>& other)
    {
        std::swap(m, other.m);
        std::swap(n, other.n);
        std::swap(data, other.data);
    }

private:
    size_t m;
    size_t n;
    std::vector<T> data;
};

template <typename T>
class Array3D
{
public:
    Array3D(size_t n1, size_t n2, size_t n3)
        : shape(3), data(n1 * n2 * n3)
    {
        shape[0] = n1;
        shape[1] = n2;
        shape[2] = n3;
    }

    size_t dim(size_t i) const    { return shape[i]; }

    T& operator()(size_t i, size_t j, size_t k)
    {
        return data[(i*shape[1] + j) * shape[2] + k];
    }
    const T& operator()(size_t i, size_t j, size_t k) const
    {
        return data[(i*shape[1] + j) * shape[2] + k];
    }

    void fill(T value)
    {
        std::fill(data.begin(), data.end(), value);
    }

    void add_cross(const std::vector<T>& x, const Array2D<T>& yz)
    {
        assert(x.size()  == dim(0));
        assert(yz.rows() == dim(1));
        assert(yz.cols() == dim(2));

        for (size_t i = 0; i < dim(0); ++i)
            for (size_t j = 0; j < dim(1); ++j)
                for (size_t k = 0; k < dim(2); ++k)
                    (*this)(i, j, k) += x[i] * yz(j, k);
    }

private:
    std::vector<size_t> shape;
    std::vector<T> data;
};


////////////////////////////////////////////////////////////////////////////////
// Abstract matrix and tensor generators
////////////////////////////////////////////////////////////////////////////////

class MatrixGenerator
{
public:
    MatrixGenerator(size_t m_, size_t n_)
        : m(m_), n(n_)
    {
    }

    size_t rows() const    { return m; }
    size_t cols() const    { return n; }

    virtual double entry(size_t i, size_t j) const = 0;

private:
    size_t m;
    size_t n;
};

class TensorGenerator
{
public:
    TensorGenerator(const std::vector<size_t>& dims_)
        : shape(dims_)
    {
    }

    TensorGenerator(size_t n0, size_t n1, size_t n2)
        : shape(3)
    {
        shape[0] = n0;
        shape[1] = n1;
        shape[2] = n2;
    }

    size_t num_dims() const { return shape.size(); }
    size_t dim(size_t i) const    { return shape[i]; }

    virtual double entry(const std::vector<size_t>& I) const = 0;

private:
    std::vector<size_t> shape;
};


class TensorSliceMatrixGenerator : public MatrixGenerator
{
public:
    TensorSliceMatrixGenerator(const TensorGenerator& X_,
                               const std::vector<size_t>& I_,
                               int axis1_, int axis2_)
        : MatrixGenerator( X_.dim(axis1_), X_.dim(axis2_) ),
          X(X_), I(I_), axis1(axis1_), axis2(axis2_)
    {
    }

    double entry(size_t i, size_t j) const
    {
        I[axis1] = i;
        I[axis2] = j;
        return X.entry(I);
    }

private:
    const TensorGenerator& X;
    mutable std::vector<size_t> I;
    int axis1, axis2;
};

////////////////////////////////////////////////////////////////////////////////
// Adaptive Cross Approximation (2D and 3D)
////////////////////////////////////////////////////////////////////////////////

// find the index of the largest entry in absolute value
size_t argmax_abs(const std::vector<double>& v)
{
    const size_t n = v.size();
    assert(n > 0);

    double max = std::abs(v[0]);
    size_t argmax = 0;

    for (size_t i = 1; i < n; ++i)
    {
        double z = std::abs(v[i]);
        if (z > max)
        {
            max = z;
            argmax = i;
        }
    }
    return argmax;
}


// max_skipcount: terminate after how many skipped rows
// max_tolcount:  terminate after how many rows below error tolerance

Array2D<double> aca(const MatrixGenerator& A, double tol=1e-10, int maxiter=100,
        int max_skipcount=3, int max_tolcount=3, int verbose=2,
        const double * startval = NULL)
{
    size_t m = A.rows(), n = A.cols();
    assert((m > 0) && (n > 0));
    std::vector<double> col(m), row(n);

    Array2D<double> X(m, n);
    if (startval)
        X.fill(startval);
    else
        X.fill(0.0);

    int skipcount = 0, tolcount = 0;
    size_t i = A.rows() / 2;

    int k = 0;
    while (true)
    {
        if (k >= maxiter)
        {
            if (verbose >= 1)
                logger << "Maximum iteration count reached; aborting ("
                    << k << " it.)" << std::endl;
            break;
        }

        // generate error row
        for (size_t l = 0; l < n; ++l)
            row[l] = A.entry(i, l) - X(i, l);

        // find maximum error in column to use as pivot
        size_t j0 = argmax_abs(row);
        double e = std::abs(row[j0]);

        if (e < 1e-15)      // skip row if it's very small
        {
            if (verbose >= 2)
                logger << "Skipping row " << i << std::endl;

            // choose a new column by random
            i = rand() % m;
            //i = (i + 1) % m;

            if (++skipcount >= max_skipcount)
            {
                if (verbose >= 1)
                    logger << "Skipped " << skipcount << " times; stopping ("
                        << k << " it.)" << std::endl;
                break;
            }
            else
            {
                continue;
            }
        }
        else if (e < tol)   // terminate if tolerance reached often enough
        {
            if (++tolcount >= max_tolcount)
            {
                if (verbose >= 1)
                    logger << "Desired tolerance reached " << tolcount <<
                        " times; stopping (" << k << " it.)" << std::endl;
                break;
            }
        }
        else    // error is large
        {
            skipcount = tolcount = 0;       // reset the counters
        }

        if (verbose >= 2)
            logger << i << '\t' << j0 << '\t' << e << std::endl;

        // generate error column (scaled)
        for (size_t l = 0; l < m; ++l)
            col[l] = (A.entry(l, j0) - X(l, j0)) / row[j0];

        // add rank 1 correction to previous approximation
        X.add_cross(col, row);
        ++k;    // count added crosses (rank)

        // find next row for pivoting
        col[i] = 0.0;   // error is now 0 there
        i = argmax_abs(col);
    }

    return X;
}

Array3D<double> aca_3d(const TensorGenerator& A, double tol=1e-10, int maxiter=100,
        int max_skipcount=3, int max_tolcount=3, int verbose=2)
{
    assert(A.num_dims() == 3);
    size_t n0 = A.dim(0), n1 = A.dim(1), n2 = A.dim(2);
    assert((n0 > 0) && (n1 > 0) && (n2 > 0));

    std::vector<double> col(n0);

    Array3D<double> X(n0, n1, n2);
    X.fill(0.0);

    std::vector<size_t> I(3);
    I[0] = n0 / 2;
    I[1] = n1 / 2;
    I[2] = n2 / 2;

    int skipcount = 0, tolcount = 0;

    int k = 0;
    while (true)
    {
        if (k >= maxiter)
        {
            if (verbose >= 1)
                logger << "Maximum iteration count reached; aborting ("
                    << k << " outer it.)" << std::endl;
            break;
        }

        // generate error column at I
        for (size_t l = 0; l < n0; ++l)
        {
            I[0] = l;
            col[l] = A.entry(I) - X(l, I[1], I[2]);
        }

        // find maximum error in column to use as pivot
        size_t i0 = argmax_abs(col);
        double e = std::abs(col[i0]);

        if (e < 1e-15)      // skip pivot if it's very small
        {
            if (verbose >= 2)
                logger << "Skipping..." << std::endl;

            // choose a new column by random
            I[1] = rand() % n1;
            I[2] = rand() % n2;
            //I[1] = (I[1] + 1) % n1;
            //I[2] = (I[2] + 2) % n2;

            if (++skipcount >= max_skipcount)
            {
                if (verbose >= 1)
                    logger << "Skipped " << skipcount << " times; stopping ("
                        << k << " outer it.)" << std::endl;
                break;
            }
            else
            {
                continue;
            }
        }
        else if (e < tol)   // terminate if tolerance reached often enough
        {
            if (++tolcount >= max_tolcount)
            {
                if (verbose >= 1)
                    logger << "Desired tolerance reached " << tolcount <<
                        " times; stopping (" << k << " outer it.)" << std::endl;
                break;
            }
        }
        else    // error is large
        {
            skipcount = tolcount = 0;       // reset the counters
        }

        I[0] = i0;
        if (verbose >= 2)
            logger << I[0] << '\t' << I[1] << '\t' << I[2] << '\t' << e << std::endl;

        // approximate slice A[i0, :, :] by 2D ACA
        Array2D<double> mat = aca(
                TensorSliceMatrixGenerator(A, I, 1, 2),
                tol, maxiter, max_skipcount, max_tolcount, std::min(verbose, 1),
                &X(i0, 0, 0)        // use current approximation as starting value
                );

        // compute error
        for (size_t i1 = 0; i1 < n1; ++i1)
            for (size_t i2 = 0; i2 < n2; ++i2)
                mat(i1, i2) -= X(i0, i1, i2);

        const double pivot = col[i0];
        for (size_t l = 0; l < n0; ++l)
            col[l] /= pivot;

        // add rank 1 correction to previous approximation
        X.add_cross(col, mat);
        ++k;    // count added crosses (rank)

        // find next row for pivoting
        mat(I[1], I[2]) = 0.0;      // error is now 0 there
        mat.argmax_abs(I[1], I[2]);
    }

    return X;
}


////////////////////////////////////////////////////////////////////////////////
// Multilevel banded matrices
////////////////////////////////////////////////////////////////////////////////

/// Compute all sequential indices which are nonzero in a square matrix of size nxn
/// with bandwidth bw.
///
/// bw must be odd and greater than zero.
std::vector<size_t> compute_banded_sparsity(size_t n, int bw)
{
    std::vector<size_t> I;
    for (int j = 0; j < n; ++j)
        for (int i = std::max(0, j-bw); i < std::min((int)n, j+bw+1); ++i)
            I.push_back(i + j*n);
    return I;
}

void inline from_seq2(size_t i, size_t bs, size_t out[2])
{
    out[0] = i / bs;
    out[1] = i % bs;
}

template <int L>
void inline reindex_from_multilevel(const size_t M[L], const size_t bs[L], size_t out[2])
{
    size_t ij[2];
    out[0] = out[1] = 0;

    for (size_t k = 0; k < L; ++k)
    {
        from_seq2(M[k], bs[k], ij);
        out[0] *= bs[k];
        out[0] += ij[0];
        out[1] *= bs[k];
        out[1] += ij[1];
    }
}

typedef double (*MatrixEntryFn)(size_t i, size_t j, void * data);

class ReorderedMatrixGenerator : public MatrixGenerator
{
public:
    ReorderedMatrixGenerator(
            MatrixEntryFn entryfunc_,
            void * data_,
            const std::vector<size_t>& sparsidx0_,
            const std::vector<size_t>& sparsidx1_,
            size_t block_sizes_[2])
        : MatrixGenerator(sparsidx0_.size(), sparsidx1_.size()),
          entryfunc(entryfunc_),
          data(data_),
          sparsidx0(sparsidx0_), sparsidx1(sparsidx1_)
    {
        block_sizes[0] = block_sizes_[0];
        block_sizes[1] = block_sizes_[1];
    }

    double entry(size_t i, size_t j) const
    {
        size_t M[2], I[2];
        M[0] = sparsidx0[i];
        M[1] = sparsidx1[j];
        reindex_from_multilevel<2>(M, block_sizes, I);

        // return entry I[0],I[1] of the original matrix
        return entryfunc(I[0], I[1], data);
    }

private:
    MatrixEntryFn entryfunc;
    void * data;
    std::vector<size_t> sparsidx0, sparsidx1;
    size_t block_sizes[2];
};

class ReorderedTensor3Generator : public TensorGenerator
{
public:
    ReorderedTensor3Generator(
            MatrixEntryFn entryfunc_,
            void * data_,
            const std::vector<size_t>& sparsidx0_,
            const std::vector<size_t>& sparsidx1_,
            const std::vector<size_t>& sparsidx2_,
            size_t block_sizes_[3])
        : TensorGenerator(sparsidx0_.size(), sparsidx1_.size(), sparsidx2_.size()),
          entryfunc(entryfunc_),
          data(data_),
          sparsidx0(sparsidx0_), sparsidx1(sparsidx1_), sparsidx2(sparsidx2_)
    {
        block_sizes[0] = block_sizes_[0];
        block_sizes[1] = block_sizes_[1];
        block_sizes[2] = block_sizes_[2];
    }

    double entry(const std::vector<size_t>& J) const
    {
        size_t M[3], I[2];
        M[0] = sparsidx0[J[0]];
        M[1] = sparsidx1[J[1]];
        M[2] = sparsidx2[J[2]];
        reindex_from_multilevel<3>(M, block_sizes, I);

        // return entry I[0],I[1] of the original matrix
        return entryfunc(I[0], I[1], data);
    }

private:
    MatrixEntryFn entryfunc;
    void * data;
    std::vector<size_t> sparsidx0, sparsidx1, sparsidx2;
    size_t block_sizes[3];
};

/// Convert a compressed, reordered 2D multilevel matrix back to standard sparse COO format
void inflate_2d(const Array2D<double>& X,
                const std::vector<size_t>& sparsidx0,
                const std::vector<size_t>& sparsidx1,
                size_t block_sizes[2],
                // output parameters
                std::vector<size_t>& entries_i,
                std::vector<size_t>& entries_j,
                std::vector<double>& entries
               )
{
    const size_t N0 = X.rows(), N1 = X.cols();

    entries_i.reserve(N0*N1);
    entries_j.reserve(N0*N1);
    entries  .reserve(N0*N1);

    size_t x[2], y[2];

    for (size_t i0 = 0; i0 < N0; ++i0)
    {
        const size_t s0 = sparsidx0[i0];
        x[0] = s0 % block_sizes[0];
        y[0] = s0 / block_sizes[0];

        for (size_t i1 = 0; i1 < N1; ++i1)
        {
            const size_t s1 = sparsidx1[i1];
            x[1] = s1 % block_sizes[1];
            y[1] = s1 / block_sizes[1];

            entries_i.push_back(x[0] * block_sizes[1] + x[1]);
            entries_j.push_back(y[0] * block_sizes[1] + y[1]);
            entries  .push_back(X(i0,i1));
        }
    }
}

/// Convert a compressed, reordered 3D multilevel matrix back to standard sparse COO format
void inflate_3d(const Array3D<double>& X,
                const std::vector<size_t>& sparsidx0,
                const std::vector<size_t>& sparsidx1,
                const std::vector<size_t>& sparsidx2,
                size_t block_sizes[3],
                // output parameters
                std::vector<size_t>& entries_i,
                std::vector<size_t>& entries_j,
                std::vector<double>& entries
               )
{
    const size_t N0 = X.dim(0), N1 = X.dim(1), N2 = X.dim(2);

    entries_i.reserve(N0*N1*N2);
    entries_j.reserve(N0*N1*N2);
    entries  .reserve(N0*N1*N2);

    size_t x[3], y[3];

    for (size_t i0 = 0; i0 < N0; ++i0)
    {
        const size_t s0 = sparsidx0[i0];
        x[0] = s0 % block_sizes[0];
        y[0] = s0 / block_sizes[0];

        for (size_t i1 = 0; i1 < N1; ++i1)
        {
            const size_t s1 = sparsidx1[i1];
            x[1] = s1 % block_sizes[1];
            y[1] = s1 / block_sizes[1];

            for (size_t i2 = 0; i2 < N2; ++i2)
            {
                const size_t s2 = sparsidx2[i2];
                x[2] = s2 % block_sizes[2];
                y[2] = s2 / block_sizes[2];

                entries_i.push_back((x[0] * block_sizes[1] + x[1]) * block_sizes[2] + x[2]);
                entries_j.push_back((y[0] * block_sizes[1] + y[1]) * block_sizes[2] + y[2]);
                entries  .push_back(X(i0,i1,i2));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Main entry points
////////////////////////////////////////////////////////////////////////////////

extern void fast_assemble_2d(
        // matrix specification
        MatrixEntryFn entryfunc,
        void * data,
        size_t n0, int bw0,
        size_t n1, int bw1,
        // ACA parameters
        double tol, int maxiter, int max_skipcount, int max_tolcount,
        int verbose,
        // output arguments
        std::vector<size_t>& entries_i,
        std::vector<size_t>& entries_j,
        std::vector<double>& entries)
{
    std::vector<size_t> sparsidx0 = compute_banded_sparsity(n0, bw0);
    std::vector<size_t> sparsidx1 = compute_banded_sparsity(n1, bw1);
    size_t block_sizes[2] = { n0, n1 };

    Array2D<double> X = aca
        (
            ReorderedMatrixGenerator(entryfunc, data,
                sparsidx0, sparsidx1, block_sizes),
            tol, maxiter, max_skipcount, max_tolcount, verbose
        );

    inflate_2d(X, sparsidx0, sparsidx1, block_sizes,
            entries_i, entries_j, entries);
}


extern void fast_assemble_3d(
        // matrix specification
        MatrixEntryFn entryfunc,
        void * data,
        size_t n0, int bw0,
        size_t n1, int bw1,
        size_t n2, int bw2,
        // ACA parameters
        double tol, int maxiter, int max_skipcount, int max_tolcount,
        int verbose,
        // output arguments
        std::vector<size_t>& entries_i,
        std::vector<size_t>& entries_j,
        std::vector<double>& entries)
{
    std::vector<size_t> sparsidx0 = compute_banded_sparsity(n0, bw0);
    std::vector<size_t> sparsidx1 = compute_banded_sparsity(n1, bw1);
    std::vector<size_t> sparsidx2 = compute_banded_sparsity(n2, bw2);
    size_t block_sizes[3] = { n0, n1, n2 };

    Array3D<double> X = aca_3d
        (
            ReorderedTensor3Generator(entryfunc, data,
                sparsidx0, sparsidx1, sparsidx2, block_sizes),
            tol, maxiter, max_skipcount, max_tolcount, verbose
        );

    inflate_3d(X, sparsidx0, sparsidx1, sparsidx2, block_sizes,
            entries_i, entries_j, entries);
}