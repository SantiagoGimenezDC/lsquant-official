#ifndef SPARSE_MATRIX_KQUANT
#define SPARSE_MATRIX_KQUANT

#include <assert.h> /* assert */
#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <fftw3.h>

#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Sparse>

#include"sparse_matrix.hpp"

using std::complex;
using std::vector;
using std::string;

typedef complex<double> type;



class SparseMatrixType_kQuant  : public SparseMatrixType
{
public:
  SparseMatrixType_kQuant()
  {}
  ~SparseMatrixType_kQuant() {}


  void Multiply(const value_t a, const value_t *x, const value_t b, value_t *y);
  void Multiply(const value_t a, const vector_t& x, const value_t b, vector_t& y);
  inline 
  void Multiply(const value_t *x, value_t *y){ Multiply(value_t(1.0,0),x,value_t(0,0),y);};
  inline 
  void Multiply(const vector_t& x, vector_t& y){ Multiply(value_t(1.0,0),x,value_t(0,0),y);};

  

  void BatchMultiply(const int batchSize, const value_t a, const value_t *x, const value_t b, value_t *y);

  
  void to_rSpace_pruned(type, const type );  
  void to_kSpace_pruned(type, const type );
private:
  Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType> matrix_;
  vector<indexType> rows_;
  vector<indexType> cols_;
  vector<complex<double> > vals_;

  fftw_complex *fft_input_;// = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * W * LE );
  fftw_complex *fft_output_;// = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * W * LE );
  fftw_plan fftw_plan_FORWD_, fftw_plan_BACK_;// = fftw_plan_dft_2d(W, LE, fft_input, fft_output, FFTW_BACKWARD, FFTW_ESTIMATE);

  
  double n_basis_, n_kPoints;

  Eigen::Vector2d A1_;
  Eigen::Vector2d A2_;
  
  Eigen::Vector2d b1_;
  Eigen::Vector2d b2_;


  Eigen::VectorXd phases_;
  Eigen::VectorXd kList_;
  std::vector<Eigen::Vector2d> nonZeroList_;

};


#endif
