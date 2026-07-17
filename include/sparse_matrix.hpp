#ifndef SPARSE_MATRIX
#define SPARSE_MATRIX

#include <assert.h> /* assert */
#include <iostream>
#include <vector>
#include <complex>
#include <fstream>

#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Sparse>

using std::complex;
using std::vector;
using std::string;

typedef long int indexType;

namespace Sparse
{
bool OPERATOR_FromCSRFile(const std::string input, int &dim, vector<indexType> &columns, vector<indexType> &rowIndex, vector<complex<double> > &values);
};

class SparseMatrixType_BASE
{
	
	
public:
	typedef complex<double> value_t;
	typedef vector< value_t > vector_t;
	
	  int numRows() { return numRows_; };
	  int numCols() { return numCols_; };
	  int rank() { return ((this->numRows() > this->numCols()) ? this->numCols() : this->numRows()); };
	  void setDimensions(const int numRows, const int numCols)
	  {
		numRows_ = numRows;
		numCols_ = numCols;
	  };
	  void SetID(string id) { id_ = id; }
	  string ID() const { return id_; }
		
	  bool isIdentity(){ return (bool)( ID()=="1"); };

private:
  int numRows_, numCols_;
  string id_;
};


// MKL LIBRARIES
class SparseMatrixType  : public SparseMatrixType_BASE
{
public:
  SparseMatrixType()
  {}
  ~SparseMatrixType() {}
    
  
  string matrixType() const { return "CSR Matrix from Eigen."; };
  virtual void Multiply(const value_t , const value_t *, const value_t , value_t * );
  virtual void Multiply(const value_t , const vector_t& , const value_t , vector_t& );
  void Rescale(const value_t a,const value_t b);

  inline void Multiply(const value_t *x, value_t *y){ Multiply(value_t(1.0,0),x,value_t(0,0),y);};
  inline void Multiply(const vector_t& x, vector_t& y){ Multiply(value_t(1.0,0),x,value_t(0,0),y);};


  void BatchMultiply(const int batchSize, const value_t a, const value_t *x, const value_t b, value_t *y);

  void ConvertFromCOO(vector<indexType> &rows, vector<indexType> &cols, vector<complex<double> > &vals);
  void ConvertFromCSR(vector<indexType> &rowIndex, vector<indexType> &cols, vector<complex<double> > &vals);

  vector<indexType>* rows() {return &rows_;};
  vector<indexType>* cols() {return &cols_;};
  vector<complex<double> >* vals(){return &vals_;};
  Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>& eigen_matrix(){return matrix_;};

  void set_eigen_matrix( Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>& new_matrix ){
    matrix_ = new_matrix;
    setDimensions(new_matrix.rows(),new_matrix.cols());
  };

  double scaleFactor(){ return a_; };
  double shiftFactor(){ return b_; };
  Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* Matrix(){return &matrix_;};
  
private:
  Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType> matrix_;
  vector<indexType> rows_;
  vector<indexType> cols_;
  vector<complex<double> > vals_;

  double a_, b_;
};



class SparseMatrixBuilder
{
public:
  void setSparseMatrix(SparseMatrixType *b)
  {
    _matrix_type = b;
  };

public:
  void BuildOPFromCSRFile(const std::string input)
  {
    vector<indexType> columns, rowIndex;
    vector<complex<double> > values;
    int dim;

    if(!Sparse::OPERATOR_FromCSRFile(input, dim, columns, rowIndex, values))
      return;
    
    _matrix_type->setDimensions(dim, dim);
    _matrix_type->ConvertFromCSR( columns,rowIndex, values);
    std::cout << "OPERATOR SUCCESSFULLY BUILT" << std::endl;
  }
  SparseMatrixType *_matrix_type;
};


// ─────────────────────────────────────────────────────────────────────────────
// SparseMatrixType_kQuant
//
// Extends SparseMatrixType with the Bloch phase factors needed to apply
// the unitary Bloch transform B and its adjoint B† on-the-fly, without
// storing the full (Nk·W)×(Nk·W) matrix.
//
// The Bloch transform factorises as (atom gauge):
//
//   B_{(ik,α),(iR,β)} = δ_{αβ} × lat_phase(ik,iR) × atom_phase(ik,α)
//
//   lat_phase(ik,iR) = φ_x[n1_ik, m1_iR]          (separable 1D phases)
//                    × φ_y[n2_ik, m2_iR]
//                    × φ_z[n3_ik, m3_iR]
//
//   φ_d[n,m] = exp(i·2π·n·m / k_d)
//
//   atom_phase(ik,α) = exp(i k_ik·τ_α) / √Nk       (already includes 1/√Nk)
//
// The overridden Multiply applies the full effective Hamiltonian:
//
//   H_eff |ψ⟩ = H_k |ψ⟩  +  B† V_disorder B |ψ⟩
//
// where H_k is the k-space sparse block-diagonal Hamiltonian (parent class)
// and V_disorder is a diagonal Anderson potential in real space.
// The B / B† transforms are executed via FFTW 3D batch DFTs:
//
//   B†:  conj(atom_phase) × ψ_k  → fftw_FORWARD  → ψ_real
//   B:   fftw_BACKWARD  →  atom_phase × result   → ψ_k
// ─────────────────────────────────────────────────────────────────────────────

#include <fftw3.h>

class SparseMatrixType_kQuant : public SparseMatrixType
{
public:
    SparseMatrixType_kQuant()
        : Nk(0), W(0), kx(0), ky(0), kz(0),
          plan_fwd(nullptr), plan_bwd(nullptr) {}

    ~SparseMatrixType_kQuant()
    {
        if (plan_fwd) { fftw_destroy_plan(plan_fwd); plan_fwd = nullptr; }
        if (plan_bwd) { fftw_destroy_plan(plan_bwd); plan_bwd = nullptr; }
    }

    // ── Dimensions ────────────────────────────────────────────────────────────
    int Nk;               // total k-points
    int W;                // orbitals per unit cell
    int kx, ky, kz;       // k-grid dimensions

    // ── Phase data ────────────────────────────────────────────────────────────
    std::vector<std::array<int,3>> n_grid;     // [Nk]  integer grid indices
    std::vector<value_t>           atom_phases; // [Nk*W] exp(i k·τ_α)/√Nk
    std::vector<value_t>           phi_x;       // [kx*kx]
    std::vector<value_t>           phi_y;       // [ky*ky]
    std::vector<value_t>           phi_z;       // [kz*kz]

    // ── Anderson disorder (diagonal in real space) ─────────────────────────
    // disorder[iR*W + α] = on-site potential at orbital α in unit cell iR
    // Real-valued physically, but stored as complex for generality.
    std::vector<value_t> disorder;

    void SetDisorder(const std::vector<value_t>& dis)
    {
        assert((int)dis.size() == Nk * W);
        disorder = dis;
    }

    // Fill disorder with uniform random values in [-amplitude/2, +amplitude/2]
    // per unit cell (same value for all W orbitals in a cell, standard Anderson)
    void GenerateAndersonDisorder(double amplitude, unsigned int seed = 42);

    // ── I/O ──────────────────────────────────────────────────────────────────
    bool ReadPhasesFromFile(const std::string& filename);

    // ── FFTW setup ────────────────────────────────────────────────────────────
    // Must be called after ReadPhasesFromFile, before any Multiply call.
    // Creates in-place FFTW plans for the (kx×ky×kz) batch DFTs.
    void PrepareFFT();

    // ── Multiply (overrides parent) ───────────────────────────────────────────
    // Computes:  y = a * (H_k + B† V B) * x + b * y
    // If disorder is empty, falls back to H_k only (pure k-space).

    virtual void Multiply(const value_t a, const value_t * vec1, const value_t b, value_t * vec2) override { this->Multiply_kQuant(a,  vec1,  b,  vec2); } ;
    virtual void Multiply(const value_t a, const vector_t& vec1, const value_t b, vector_t& vec2) override { this->Multiply_kQuant(a,  vec1,  b,  vec2); } ;

    void Multiply_kQuant(const value_t , const value_t *, const value_t , value_t * ) ;
    void Multiply_kQuant(const value_t , const vector_t& , const value_t , vector_t& );

    void Multiply_kQuant_bak(const value_t , const value_t *, const value_t , value_t * ) ;
    void Multiply_kQuant_bak(const value_t , const vector_t& , const value_t , vector_t& );

  
  //  virtual  void Multiply(const value_t , const value_t*,  const value_t ,       value_t*   ) override;
  //virtual void Multiply(const value_t , const vector_t&  , const value_t ,       vector_t&  ) override;

    // ── Bloch transforms (exposed for external use) ───────────────────────────
    void apply_B       (value_t* out, const value_t* in) const;
    void apply_Bdagger (value_t* out, const value_t* in) const;

    void apply_B_FFT( value_t* out, const value_t* in);
    void apply_Bdagger_FFT( value_t* out, const value_t* in);
  
private:
    // Working buffer for the disorder application (size Nk*W, FFTW-aligned)
    std::vector<value_t> fft_buf;
    fftw_plan            plan_fwd;   // B†: k→real  (FFTW_FORWARD)
    fftw_plan            plan_bwd;   // B:  real→k  (FFTW_BACKWARD)

    inline value_t lattice_phase(int ik, int iR) const
    {
        return phi_x[ n_grid[ik][0]*kx + n_grid[iR][0] ]
             * phi_y[ n_grid[ik][1]*ky + n_grid[iR][1] ]
             * phi_z[ n_grid[ik][2]*kz + n_grid[iR][2] ];
    }
};









class SparseMatrixType_kQuant_nonOrth : public SparseMatrixType
{
public:
    SparseMatrixType_kQuant_nonOrth()
        : Nk(0), W(0), kx(0), ky(0), kz(0),
          plan_fwd(nullptr), plan_bwd(nullptr) {}

    ~SparseMatrixType_kQuant_nonOrth()
    {
        if (plan_fwd) { fftw_destroy_plan(plan_fwd); plan_fwd = nullptr; }
        if (plan_bwd) { fftw_destroy_plan(plan_bwd); plan_bwd = nullptr; }
    }

    // ── Dimensions ────────────────────────────────────────────────────────────
    int Nk;               // total k-points
    int W;                // orbitals per unit cell
    int kx, ky, kz;       // k-grid dimensions

    // ── Phase data ────────────────────────────────────────────────────────────
    std::vector<std::array<int,3>> n_grid;     // [Nk]  integer grid indices
    std::vector<value_t>           atom_phases; // [Nk*W] exp(i k·τ_α)/√Nk
    std::vector<value_t>           phi_x;       // [kx*kx]
    std::vector<value_t>           phi_y;       // [ky*ky]
    std::vector<value_t>           phi_z;       // [kz*kz]

    // ── Anderson disorder (diagonal in real space) ─────────────────────────
    // disorder[iR*W + α] = on-site potential at orbital α in unit cell iR
    // Real-valued physically, but stored as complex for generality.
    std::vector<value_t> disorder;

    void SetDisorder(const std::vector<value_t>& dis)
    {
        assert((int)dis.size() == Nk * W);
        disorder = dis;
    }

    // Fill disorder with uniform random values in [-amplitude/2, +amplitude/2]
    // per unit cell (same value for all W orbitals in a cell, standard Anderson)
    void GenerateAndersonDisorder(double amplitude, unsigned int seed = 42);

    // ── I/O ──────────────────────────────────────────────────────────────────
    bool ReadPhasesFromFile(const std::string& filename);

    // ── FFTW setup ────────────────────────────────────────────────────────────
    // Must be called after ReadPhasesFromFile, before any Multiply call.
    // Creates in-place FFTW plans for the (kx×ky×kz) batch DFTs.
    void PrepareFFT();

    // ── Multiply (overrides parent) ───────────────────────────────────────────
    // Computes:  y = a * (H_k + B† V B) * x + b * y
    // If disorder is empty, falls back to H_k only (pure k-space).


    void Hk_clean_nonOrth(const value_t *, value_t * ) ;
    void vel_i_nonOrth(const value_t *, value_t *, int ) ;

    virtual void Multiply(const value_t a, const value_t * vec1, const value_t b, value_t * vec2) override { this->Multiply_kQuant(a,  vec1,  b,  vec2); } ;
    virtual  void Multiply(const value_t a, const vector_t& vec1, const value_t b, vector_t& vec2) override { this->Multiply_kQuant(a,  vec1,  b,  vec2); } ;

    void Multiply_kQuant(const value_t , const value_t *, const value_t , value_t * ) ;
    void Multiply_kQuant(const value_t , const vector_t& , const value_t , vector_t& );

    // ── Bloch transforms (exposed for external use) ───────────────────────────
    void apply_B       (value_t* out, const value_t* in) const;
    void apply_Bdagger (value_t* out, const value_t* in) const;

    void apply_B_FFT( value_t* out, const value_t* in);
    void apply_Bdagger_FFT( value_t* out, const value_t* in);

    void set_S(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_S){    Sk_ = new_S;    }

    void set_Hk(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_Hk){  Hk_ = new_Hk;   }
    void set_dHk_1(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_dHk_1){  dHk_1_ = new_dHk_1;   }
    void set_dHk_2(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_dHk_2){  dHk_2_ = new_dHk_2;   }
    void set_dSk_1(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_dSk_1){  dSk_1_ = new_dSk_1;   }
    void set_dSk_2(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_dSk_2){  dSk_2_ = new_dSk_2;   }



  
private:
  Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* Hk_, *Sk_, *dHk_1_, *dHk_2_, *dSk_1_, *dSk_2_;


    // Working buffer for the disorder application (size Nk*W, FFTW-aligned)
    std::vector<value_t> fft_buf;
    fftw_plan            plan_fwd;   // B†: k→real  (FFTW_FORWARD)
    fftw_plan            plan_bwd;   // B:  real→k  (FFTW_BACKWARD)

    inline value_t lattice_phase(int ik, int iR) const
    {
        return phi_x[ n_grid[ik][0]*kx + n_grid[iR][0] ]
             * phi_y[ n_grid[ik][1]*ky + n_grid[iR][1] ]
             * phi_z[ n_grid[ik][2]*kz + n_grid[iR][2] ];
    }
};













class SparseMatrixType_kQuant_nonOrth_ChrisVel : public SparseMatrixType
{
public:
    SparseMatrixType_kQuant_nonOrth_ChrisVel()
        : Nk(0), W(0), kx(0), ky(0), kz(0),
          plan_fwd(nullptr), plan_bwd(nullptr) {}

    ~SparseMatrixType_kQuant_nonOrth_ChrisVel()
    {
        if (plan_fwd) { fftw_destroy_plan(plan_fwd); plan_fwd = nullptr; }
        if (plan_bwd) { fftw_destroy_plan(plan_bwd); plan_bwd = nullptr; }
    }

    // ── Dimensions ────────────────────────────────────────────────────────────
    int Nk;               // total k-points
    int W;                // orbitals per unit cell
    int kx, ky, kz;       // k-grid dimensions

    // ── Phase data ────────────────────────────────────────────────────────────
    std::vector<std::array<int,3>> n_grid;     // [Nk]  integer grid indices
    std::vector<value_t>           atom_phases; // [Nk*W] exp(i k·τ_α)/√Nk
    std::vector<value_t>           phi_x;       // [kx*kx]
    std::vector<value_t>           phi_y;       // [ky*ky]
    std::vector<value_t>           phi_z;       // [kz*kz]

    // ── Anderson disorder (diagonal in real space) ─────────────────────────
    // disorder[iR*W + α] = on-site potential at orbital α in unit cell iR
    // Real-valued physically, but stored as complex for generality.
    std::vector<value_t> disorder;

    void SetDisorder(const std::vector<value_t>& dis)
    {
        assert((int)dis.size() == Nk * W);
        disorder = dis;
    }

    // Fill disorder with uniform random values in [-amplitude/2, +amplitude/2]
    // per unit cell (same value for all W orbitals in a cell, standard Anderson)
    void GenerateAndersonDisorder(double amplitude, unsigned int seed = 42);

    // ── I/O ──────────────────────────────────────────────────────────────────
    bool ReadPhasesFromFile(const std::string& filename);

    // ── FFTW setup ────────────────────────────────────────────────────────────
    // Must be called after ReadPhasesFromFile, before any Multiply call.
    // Creates in-place FFTW plans for the (kx×ky×kz) batch DFTs.
    void PrepareFFT();

    // ── Multiply (overrides parent) ───────────────────────────────────────────
    // Computes:  y = a * (H_k + B† V B) * x + b * y
    // If disorder is empty, falls back to H_k only (pure k-space).


    void Hk_clean_nonOrth(const value_t *, value_t * ) ;
    void vel_i_nonOrth(const value_t *, value_t *, int ) ;

    virtual void Multiply(const value_t a, const value_t * vec1, const value_t b, value_t * vec2) override { this->Multiply_kQuant(a,  vec1,  b,  vec2); } ;
    virtual  void Multiply(const value_t a, const vector_t& vec1, const value_t b, vector_t& vec2) override { this->Multiply_kQuant(a,  vec1,  b,  vec2); } ;

    void Multiply_kQuant(const value_t , const value_t *, const value_t , value_t * ) ;
    void Multiply_kQuant(const value_t , const vector_t& , const value_t , vector_t& );

    // ── Bloch transforms (exposed for external use) ───────────────────────────
    void apply_B       (value_t* out, const value_t* in) const;
    void apply_Bdagger (value_t* out, const value_t* in) const;

    void apply_B_FFT( value_t* out, const value_t* in);
    void apply_Bdagger_FFT( value_t* out, const value_t* in);

    void set_S(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_S){    Sk_ = new_S;    }

    void set_Hk(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_Hk){  Hk_ = new_Hk;   }
    void set_dHk_1(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_dHk_1){  dHk_1_ = new_dHk_1;   }
    void set_dHk_2(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_dHk_2){  dHk_2_ = new_dHk_2;   }
  
    void set_A_1(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_A_1){  A_1_ = new_A_1;  (*A_1d_)=(*A_1_).conjugate().transpose();   }
    void set_A_2(Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* new_A_2){  A_2_ = new_A_2;  (*A_2d_)=(*A_2_).conjugate().transpose();  }



  
private:
  Eigen::SparseMatrix<complex<double>, Eigen::RowMajor, indexType>* Hk_, *Sk_, *dHk_1_, *dHk_2_, *A_1_, *A_2_, *A_1d_, *A_2d_;


    // Working buffer for the disorder application (size Nk*W, FFTW-aligned)
    std::vector<value_t> fft_buf;
    fftw_plan            plan_fwd;   // B†: k→real  (FFTW_FORWARD)
    fftw_plan            plan_bwd;   // B:  real→k  (FFTW_BACKWARD)

    inline value_t lattice_phase(int ik, int iR) const
    {
        return phi_x[ n_grid[ik][0]*kx + n_grid[iR][0] ]
             * phi_y[ n_grid[ik][1]*ky + n_grid[iR][1] ]
             * phi_z[ n_grid[ik][2]*kz + n_grid[iR][2] ];
    }
};



#endif
