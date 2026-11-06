#include "fftw_wrapper.hpp"
#include "Kubo_solver_FFT.hpp"
#include <vector>
#include <algorithm>

namespace chebyshev{

void Kubo_solver_FFT::Bastin_FFTs(Vectors_sliced &chebVecL, Vectors_sliced &chebVecR, value_t r_data[], int s){

  const std::complex<double> im(0,1);

  
  size_t size = section_size_;
  
  if( s != num_sections_-1 )
    size -= Hamiltonian_dummyMoms_.SystemSize() % num_sections_;


  
  value_t pre_factors [ nump_ ];      
      
  for(int m = 0; m < M_; m++)
    pre_factors[m]  = value_t( ( 2 - ( m == 0 ) ) * Jackson_kernel(m, M_) ) * std::polar( 1.0, M_PI * m / ( 2.0 * nump_ ) ) ;




#pragma omp parallel 
  {
    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;

    if (id == Nthrds-1)
      l_end = size;
  
  
  //8 plans + 14 [M] sized vectors per thread. Can it be reduced to 2 plans and some 4 vectors?

    out_of_place_dft
      re_bras( nump_, BACKWARD ),
      im_bras( nump_, BACKWARD ),
      
      re_kets( nump_, BACKWARD ),
      im_kets( nump_, BACKWARD ),

     
      re_D_bras( nump_, BACKWARD ),
      im_D_bras( nump_, BACKWARD ),
      
      re_D_kets( nump_, BACKWARD ),
      im_D_kets( nump_, BACKWARD );


   std::complex<double> bra, ket, //variables to access values within bras/kets         
                        p[nump_], w[nump_]; //Dot product partial results;

   for(int k = 0; k < nump_; k++){
     p[k] = 0;
     w[k] = 0;
   }
    

    //FFTW plans. All plans are in-place backward 1D FFTS of the entry variables; These functions are no thread safe for some reason, hence the # pragma critical
# pragma omp critical
   {
     re_bras.create();
     im_bras.create();
     re_kets.create();
     im_kets.create();
     
     re_D_bras.create();
     im_D_bras.create();
     re_D_kets.create();
     im_D_kets.create();
   }

   
   //VectorList< Moments::value_t >  chebMu_L = chebVecL.Chebmu();
   //VectorList< Moments::value_t >  chebMu_R = chebVecR.Chebmu();

   
   for(int l = l_start; l < l_end; l++){
     for(int m = 0; m < M_; m++){
	//The following casting allows to simplify all template instantiations: converts all to double.
	//All operations here are performed in double precision.
	//I also don`t think fftw supports long double as of right now.
	//The row-wise accessed of the col-major matrices are some of the longest operations in this loop!!

       std::complex<r_value_t> im_m = static_cast<std::complex<r_value_t>> (m);
	
       bra = chebVecL.Vector(m)[l]; 

       //Bra Greens functions FFT inputs:
       re_bras.input()[m] = pre_factors[m] * real( bra );
       im_bras.input()[m] = pre_factors[m] * imag( bra );
	  
	
       //The same inputs, multiplied by m, serve as inputs for the derivative FFT: 
       re_D_bras.input()[m] = im_m * re_bras.input()[m];
       im_D_bras.input()[m] = im_m * im_bras.input()[m];
	
       

	
       ket = chebVecR.Vector(m)[l];

       //Bra Greens functions FFT inputs:
       re_kets.input()[m] = pre_factors[m] * real( ket );
       im_kets.input()[m] = pre_factors[m] * imag( ket );
	  
	
       //The same inputs, multiplied by m, serve as inputs for the derivative FFT: 
       re_D_kets.input()[m] = im_m * re_kets.input()[ m ];
       im_D_kets.input()[m] = im_m * im_kets.input()[ m ];	
     }

     re_bras.execute();
     im_bras.execute();
     re_kets.execute();
     im_kets.execute();  

     re_D_bras.execute();
     im_D_bras.execute();
     re_D_kets.execute();
     im_D_kets.execute();
  



     //As mentioned previously, variable change as k=2j; This loop updates the partial results of the dot products p(E) and w(E);
     //This also performs the conjugation from bra.cdot(ket) of the complex random vector dot product.
     for(int j = 0; j < nump_/2; j++){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] += ( real( re_bras(j) ) - im * real( im_bras(j) ) ) *  //Re(G(k)) *
	         (       re_kets(j)   + im *       im_kets(j) ) + //G(k)+
 	  
	         ( conj( re_bras(j) ) - im * conj( im_bras(j) ) ) * //G(k)
	         ( real( re_kets(j) ) + im * real( im_kets(j) ) );   //Re(G(k))
	

       //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
       w[j] += ( conj( re_D_bras(j) ) - im * conj( im_D_bras(j) ) )  * //dG(k)
		 ( real( re_kets(j)   ) + im * real( im_kets(j) ) ) - //Re(G(k))
	  
		 ( real( re_bras(j) )   - im * real( im_bras(j) ) )  *  //Re(G(k))
	         ( re_D_kets(j)         + im * im_D_kets(j) ); //dG(k)
     }

     
     for(int j = nump_ / 2; j < nump_; j++){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] += ( real( re_bras(j) ) - im * real( im_bras(j) ) ) *  //Re(G(k)) *

	       ( conj(re_kets(j) )   + im *       conj( im_kets(j) ) ) + //G(k)+
 	  
	         ( re_bras(j)   - im * im_bras(j)  ) * //G(k)
	         ( real( re_kets(j) ) + im * real( im_kets(j) ) );   //Re(G(k))
	

         //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
         w[j] += (  re_D_bras(j)  - im *  im_D_bras(j)  )  * //dG(k)
		 ( real( re_kets(j)   ) + im * real( im_kets(j) ) ) - //Re(G(k))
	  
		 ( real( re_bras(j) )   - im * real( im_bras(j) ) )  *  //Re(G(k))
	         ( conj( re_D_kets(j) )         + im * conj ( im_D_kets(j) ) ); //dG(k)

     }
   }      
    
# pragma omp critical
   {
     for(int k = 0; k < nump_; k++){
       r_data[ k ]         += p[ k ];
       r_data[ k + nump_ ] += w[ k ];	
     }
   } 
 }
}










void Kubo_solver_FFT_kQuant_nonOrth::Bastin_FFTs(Vectors_sliced_kQuant_nonOrth &chebVecL, Vectors_sliced_kQuant_nonOrth &chebVecR, value_t r_data[], int s){

  const std::complex<double> im(0,1);

  
  size_t size = section_size_;
  
  if( s != num_sections_-1 )
    size -= Hamiltonian_dummyMoms_.SystemSize() % num_sections_;


  
  value_t pre_factors [ nump_ ];      
      
  for(int m = 0; m < M_; m++)
    pre_factors[m]  = value_t( ( 2 - ( m == 0 ) ) * Jackson_kernel(m, M_) ) * std::polar( 1.0, M_PI * m / ( 2.0 * nump_ ) ) ;




#pragma omp parallel 
  {
    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;

    if (id == Nthrds-1)
      l_end = size;
  
  
  //8 plans + 14 [M] sized vectors per thread. Can it be reduced to 2 plans and some 4 vectors?

    out_of_place_dft
      re_bras( nump_, BACKWARD ),
      im_bras( nump_, BACKWARD ),
      
      re_kets( nump_, BACKWARD ),
      im_kets( nump_, BACKWARD ),

     
      re_D_bras( nump_, BACKWARD ),
      im_D_bras( nump_, BACKWARD ),
      
      re_D_kets( nump_, BACKWARD ),
      im_D_kets( nump_, BACKWARD );


   std::complex<double> bra, ket, //variables to access values within bras/kets         
                        p[nump_], w[nump_]; //Dot product partial results;

   for(int k = 0; k < nump_; k++){
     p[k] = 0;
     w[k] = 0;
   }
    

    //FFTW plans. All plans are in-place backward 1D FFTS of the entry variables; These functions are no thread safe for some reason, hence the # pragma critical
# pragma omp critical
   {
     re_bras.create();
     im_bras.create();
     re_kets.create();
     im_kets.create();
     
     re_D_bras.create();
     im_D_bras.create();
     re_D_kets.create();
     im_D_kets.create();
   }

   
   //VectorList< Moments::value_t >  chebMu_L = chebVecL.Chebmu();
   //VectorList< Moments::value_t >  chebMu_R = chebVecR.Chebmu();

   
   for(int l = l_start; l < l_end; l++){
     for(int m = 0; m < M_; m++){
	//The following casting allows to simplify all template instantiations: converts all to double.
	//All operations here are performed in double precision.
	//I also don`t think fftw supports long double as of right now.
	//The row-wise accessed of the col-major matrices are some of the longest operations in this loop!!

       std::complex<r_value_t> im_m = static_cast<std::complex<r_value_t>> (m);
	
       bra = chebVecL.Vector(m)[l]; 

       //Bra Greens functions FFT inputs:
       re_bras.input()[m] = pre_factors[m] * real( bra );
       im_bras.input()[m] = pre_factors[m] * imag( bra );
	  
	
       //The same inputs, multiplied by m, serve as inputs for the derivative FFT: 
       re_D_bras.input()[m] = im_m * re_bras.input()[m];
       im_D_bras.input()[m] = im_m * im_bras.input()[m];
	
       

	
       ket = chebVecR.Vector(m)[l];

       //Bra Greens functions FFT inputs:
       re_kets.input()[m] = pre_factors[m] * real( ket );
       im_kets.input()[m] = pre_factors[m] * imag( ket );
	  
	
       //The same inputs, multiplied by m, serve as inputs for the derivative FFT: 
       re_D_kets.input()[m] = im_m * re_kets.input()[ m ];
       im_D_kets.input()[m] = im_m * im_kets.input()[ m ];	
     }

     re_bras.execute();
     im_bras.execute();
     re_kets.execute();
     im_kets.execute();  

     re_D_bras.execute();
     im_D_bras.execute();
     re_D_kets.execute();
     im_D_kets.execute();
  



     //As mentioned previously, variable change as k=2j; This loop updates the partial results of the dot products p(E) and w(E);
     //This also performs the conjugation from bra.cdot(ket) of the complex random vector dot product.
     for(int j = 0; j < nump_/2; j++){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] += ( real( re_bras(j) ) - im * real( im_bras(j) ) ) *  //Re(G(k)) *
	         (       re_kets(j)   + im *       im_kets(j) ) + //G(k)+
 	  
	         ( conj( re_bras(j) ) - im * conj( im_bras(j) ) ) * //G(k)
	         ( real( re_kets(j) ) + im * real( im_kets(j) ) );   //Re(G(k))
	

       //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
       w[j] += ( conj( re_D_bras(j) ) - im * conj( im_D_bras(j) ) )  * //dG(k)
		 ( real( re_kets(j)   ) + im * real( im_kets(j) ) ) - //Re(G(k))
	  
		 ( real( re_bras(j) )   - im * real( im_bras(j) ) )  *  //Re(G(k))
	         ( re_D_kets(j)         + im * im_D_kets(j) ); //dG(k)
     }

     
     for(int j = nump_ / 2; j < nump_; j++){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] += ( real( re_bras(j) ) - im * real( im_bras(j) ) ) *  //Re(G(k)) *

	       ( conj(re_kets(j) )   + im *       conj( im_kets(j) ) ) + //G(k)+
 	  
	         ( re_bras(j)   - im * im_bras(j)  ) * //G(k)
	         ( real( re_kets(j) ) + im * real( im_kets(j) ) );   //Re(G(k))
	

         //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
         w[j] += (  re_D_bras(j)  - im *  im_D_bras(j)  )  * //dG(k)
		 ( real( re_kets(j)   ) + im * real( im_kets(j) ) ) - //Re(G(k))
	  
		 ( real( re_bras(j) )   - im * real( im_bras(j) ) )  *  //Re(G(k))
	         ( conj( re_D_kets(j) )         + im * conj ( im_D_kets(j) ) ); //dG(k)

     }
   }      
    
# pragma omp critical
   {
     for(int k = 0; k < nump_; k++){
       r_data[ k ]         += p[ k ];
       r_data[ k + nump_ ] += w[ k ];	
     }
   } 
 }
}

















void Kubo_solver_FFT_kQuant::Bastin_FFTs(Vectors_sliced_kQuant &chebVecL, Vectors_sliced_kQuant &chebVecR, value_t r_data[], int s){

  typedef std::complex<double> cplx;

  // Rows in this section. (The old formula used Hamiltonian_dummyMoms_.SystemSize(),
  // which is never set, so it was only right for num_sections_ == 1.)
  const size_t size = ( s == num_sections_ - 1 ) ? chebVecL.LastSectionSize()
                                                 : chebVecL.SectionSize();
  const int M = M_, N = nump_;

  std::vector<cplx> pre_factors( M );
  for(int m = 0; m < M; m++)
    pre_factors[m] = cplx( ( 2 - ( m == 0 ) ) * Jackson_kernel(m, M) ) * std::polar( 1.0, M_PI * m / ( 2.0 * N ) );

  // Raw pointers to the stored Chebyshev vectors (skips VectorList indirection).
  std::vector<const cplx*> bras( M ), kets( M );
  for(int m = 0; m < M; m++){
    bras[m] = chebVecL.Vector(m).data();
    kets[m] = chebVecR.Vector(m).data();
  }

  // The 8 transforms per row are done as one batched plan (howmany = 8,
  // contiguous blocks of length N). Built once with FFTW_MEASURE and reused
  // for every section / random vector; threads run it on their own buffers
  // through fftw_execute_dft, which is thread safe. Planning itself is not,
  // so it happens here, outside the parallel region.
  //   block 0: re_bras   1: im_bras   2: re_kets   3: im_kets
  //   block 4: re_D_bras 5: im_D_bras 6: re_D_kets 7: im_D_kets
  const int NB = 8;
  if( bastin_plan_ == nullptr || bastin_plan_nump_ != N ){
    if( bastin_plan_ ) fftw_destroy_plan( bastin_plan_ );
    fftw_complex *tin  = fftw_alloc_complex( size_t(NB) * N ),
                 *tout = fftw_alloc_complex( size_t(NB) * N );
    int n[1] = { N };
    bastin_plan_ = fftw_plan_many_dft( 1, n, NB,
                                       tin,  nullptr, 1, N,
                                       tout, nullptr, 1, N,
                                       FFTW_BACKWARD, FFTW_MEASURE | FFTW_PRESERVE_INPUT );
    fftw_free( tin );
    fftw_free( tout );
    bastin_plan_nump_ = N;
  }
  const fftw_plan plan = bastin_plan_;


#pragma omp parallel
  {
    // Per-thread workspace. Inputs beyond m >= M stay zero (zero padding).
    fftw_complex *in_raw  = fftw_alloc_complex( size_t(NB) * N ),
                 *out_raw = fftw_alloc_complex( size_t(NB) * N );
    cplx *in  = reinterpret_cast<cplx*>( in_raw  ),
         *out = reinterpret_cast<cplx*>( out_raw );
    std::fill( in, in + size_t(NB) * N, cplx(0.0) );

    cplx *in_RB  = in,         *in_IB  = in + 1*N, *in_RK  = in + 2*N, *in_IK  = in + 3*N,
         *in_DRB = in + 4*N,   *in_DIB = in + 5*N, *in_DRK = in + 6*N, *in_DIK = in + 7*N;
    const cplx *RB  = out,       *IB  = out + 1*N, *RK  = out + 2*N, *IK  = out + 3*N,
               *DRB = out + 4*N, *DIB = out + 5*N, *DRK = out + 6*N, *DIK = out + 7*N;

    // Dot-product partial results p(E), w(E), accumulated in real arithmetic.
    std::vector<double> p_re( N, 0.0 ), p_im( N, 0.0 ), w_re( N, 0.0 ), w_im( N, 0.0 );

#pragma omp for schedule(static)
    for(long l = 0; l < (long)size; l++){

      for(int m = 0; m < M; m++){
        const cplx pf  = pre_factors[m],
                   bra = bras[m][l],
                   ket = kets[m][l];
        const double dm = m;

        // complex * real scalars only: no __muldc3 calls.
        in_RB[m]  = pf * bra.real();
        in_IB[m]  = pf * bra.imag();
        in_RK[m]  = pf * ket.real();
        in_IK[m]  = pf * ket.imag();

        // The same inputs, multiplied by m, feed the derivative FFTs:
        in_DRB[m] = dm * in_RB[m];
        in_DIB[m] = dm * in_IB[m];
        in_DRK[m] = dm * in_RK[m];
        in_DIK[m] = dm * in_IK[m];
      }

      fftw_execute_dft( plan, in_raw, out_raw );

      // Same algebra as the previous std::complex version, in real arithmetic:
      //   p(k) += Re(G_bra) * G_ket  +  G_bra * Re(G_ket)
      //   w(k) += dG_bra * Re(G_ket) -  Re(G_bra) * dG_ket
      // with  LB  = Re(RB) - i Re(IB)   (Re(G) of the bra)
      //       RKr = Re(RK) + i Re(IK)   (Re(G) of the ket)
      // The conjugations on G and dG switch side between j < N/2 and j >= N/2.
      for(int j = 0; j < N; j++){
        const double rb_r = RB[j].real(), rb_i = RB[j].imag(),
                     ib_r = IB[j].real(), ib_i = IB[j].imag(),
                     rk_r = RK[j].real(), rk_i = RK[j].imag(),
                     ik_r = IK[j].real(), ik_i = IK[j].imag(),
                     drb_r = DRB[j].real(), drb_i = DRB[j].imag(),
                     dib_r = DIB[j].real(), dib_i = DIB[j].imag(),
                     drk_r = DRK[j].real(), drk_i = DRK[j].imag(),
                     dik_r = DIK[j].real(), dik_i = DIK[j].imag();

        const double lb_r = rb_r, lb_i = -ib_r;    // LB
        const double kr_r = rk_r, kr_i =  ik_r;    // RKr

        double gk_r, gk_i, gb_r, gb_i, dgb_r, dgb_i, dgk_r, dgk_i;
        if( j < N / 2 ){
          gk_r  = rk_r - ik_i;    gk_i  = rk_i + ik_r;      // RK + i IK
          gb_r  = rb_r - ib_i;    gb_i  = -rb_i - ib_r;     // conj(RB) - i conj(IB)
          dgb_r = drb_r - dib_i;  dgb_i = -drb_i - dib_r;   // conj(DRB) - i conj(DIB)
          dgk_r = drk_r - dik_i;  dgk_i = drk_i + dik_r;    // DRK + i DIK
        }
        else{
          gk_r  = rk_r + ik_i;    gk_i  = -rk_i + ik_r;     // conj(RK) + i conj(IK)
          gb_r  = rb_r + ib_i;    gb_i  = rb_i - ib_r;      // RB - i IB
          dgb_r = drb_r + dib_i;  dgb_i = drb_i - dib_r;    // DRB - i DIB
          dgk_r = drk_r + dik_i;  dgk_i = -drk_i + dik_r;   // conj(DRK) + i conj(DIK)
        }

        // p += LB*Gk + Gb*RKr
        p_re[j] += ( lb_r * gk_r - lb_i * gk_i ) + ( gb_r * kr_r - gb_i * kr_i );
        p_im[j] += ( lb_r * gk_i + lb_i * gk_r ) + ( gb_r * kr_i + gb_i * kr_r );

        // w += dGb*RKr - LB*dGk
        w_re[j] += ( dgb_r * kr_r - dgb_i * kr_i ) - ( lb_r * dgk_r - lb_i * dgk_i );
        w_im[j] += ( dgb_r * kr_i + dgb_i * kr_r ) - ( lb_r * dgk_i + lb_i * dgk_r );
      }
    }

# pragma omp critical
    {
      for(int k = 0; k < N; k++){
        r_data[ k ]     += cplx( p_re[k], p_im[k] );
        r_data[ k + N ] += cplx( w_re[k], w_im[k] );
      }
    }

    fftw_free( in_raw );
    fftw_free( out_raw );
  }
}






void Kubo_solver_FFT_kQuant_nonOrth_ChrisVel::Bastin_FFTs(Vectors_sliced_kQuant_nonOrth_ChrisVel &chebVecL, Vectors_sliced_kQuant_nonOrth_ChrisVel &chebVecR, value_t r_data[], int s){

  const std::complex<double> im(0,1);

  
  size_t size = section_size_;
  
  if( s != num_sections_-1 )
    size -= Hamiltonian_dummyMoms_.SystemSize() % num_sections_;


  
  value_t pre_factors [ nump_ ];      
      
  for(int m = 0; m < M_; m++)
    pre_factors[m]  = value_t( ( 2 - ( m == 0 ) ) * Jackson_kernel(m, M_) ) * std::polar( 1.0, M_PI * m / ( 2.0 * nump_ ) ) ;




#pragma omp parallel 
  {
    int id,  Nthrds, l_start, l_end;
    id      = omp_get_thread_num();
    Nthrds  = omp_get_num_threads();
    l_start = id * size / Nthrds;
    l_end   = (id+1) * size / Nthrds;

    if (id == Nthrds-1)
      l_end = size;
  
  
  //8 plans + 14 [M] sized vectors per thread. Can it be reduced to 2 plans and some 4 vectors?

    out_of_place_dft
      re_bras( nump_, BACKWARD ),
      im_bras( nump_, BACKWARD ),
      
      re_kets( nump_, BACKWARD ),
      im_kets( nump_, BACKWARD ),

     
      re_D_bras( nump_, BACKWARD ),
      im_D_bras( nump_, BACKWARD ),
      
      re_D_kets( nump_, BACKWARD ),
      im_D_kets( nump_, BACKWARD );


   std::complex<double> bra, ket, //variables to access values within bras/kets         
                        p[nump_], w[nump_]; //Dot product partial results;

   for(int k = 0; k < nump_; k++){
     p[k] = 0;
     w[k] = 0;
   }
    

    //FFTW plans. All plans are in-place backward 1D FFTS of the entry variables; These functions are no thread safe for some reason, hence the # pragma critical
# pragma omp critical
   {
     re_bras.create();
     im_bras.create();
     re_kets.create();
     im_kets.create();
     
     re_D_bras.create();
     im_D_bras.create();
     re_D_kets.create();
     im_D_kets.create();
   }

   
   //VectorList< Moments::value_t >  chebMu_L = chebVecL.Chebmu();
   //VectorList< Moments::value_t >  chebMu_R = chebVecR.Chebmu();

   
   for(int l = l_start; l < l_end; l++){
     for(int m = 0; m < M_; m++){
	//The following casting allows to simplify all template instantiations: converts all to double.
	//All operations here are performed in double precision.
	//I also don`t think fftw supports long double as of right now.
	//The row-wise accessed of the col-major matrices are some of the longest operations in this loop!!

       std::complex<r_value_t> im_m = static_cast<std::complex<r_value_t>> (m);
	
       bra = chebVecL.Vector(m)[l]; 

       //Bra Greens functions FFT inputs:
       re_bras.input()[m] = pre_factors[m] * real( bra );
       im_bras.input()[m] = pre_factors[m] * imag( bra );
	  
	
       //The same inputs, multiplied by m, serve as inputs for the derivative FFT: 
       re_D_bras.input()[m] = im_m * re_bras.input()[m];
       im_D_bras.input()[m] = im_m * im_bras.input()[m];
	
       

	
       ket = chebVecR.Vector(m)[l];

       //Bra Greens functions FFT inputs:
       re_kets.input()[m] = pre_factors[m] * real( ket );
       im_kets.input()[m] = pre_factors[m] * imag( ket );
	  
	
       //The same inputs, multiplied by m, serve as inputs for the derivative FFT: 
       re_D_kets.input()[m] = im_m * re_kets.input()[ m ];
       im_D_kets.input()[m] = im_m * im_kets.input()[ m ];	
     }

     re_bras.execute();
     im_bras.execute();
     re_kets.execute();
     im_kets.execute();  

     re_D_bras.execute();
     im_D_bras.execute();
     re_D_kets.execute();
     im_D_kets.execute();
  



     //As mentioned previously, variable change as k=2j; This loop updates the partial results of the dot products p(E) and w(E);
     //This also performs the conjugation from bra.cdot(ket) of the complex random vector dot product.
     for(int j = 0; j < nump_/2; j++){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] += ( real( re_bras(j) ) - im * real( im_bras(j) ) ) *  //Re(G(k)) *
	         (       re_kets(j)   + im *       im_kets(j) ) + //G(k)+
 	  
	         ( conj( re_bras(j) ) - im * conj( im_bras(j) ) ) * //G(k)
	         ( real( re_kets(j) ) + im * real( im_kets(j) ) );   //Re(G(k))
	

       //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
       w[j] += ( conj( re_D_bras(j) ) - im * conj( im_D_bras(j) ) )  * //dG(k)
		 ( real( re_kets(j)   ) + im * real( im_kets(j) ) ) - //Re(G(k))
	  
		 ( real( re_bras(j) )   - im * real( im_bras(j) ) )  *  //Re(G(k))
	         ( re_D_kets(j)         + im * im_D_kets(j) ); //dG(k)
     }

     
     for(int j = nump_ / 2; j < nump_; j++){
       //Here: p(k) += Re(G(k)) * G(k) + G(k) * Re(G(k)).
       p[j] += ( real( re_bras(j) ) - im * real( im_bras(j) ) ) *  //Re(G(k)) *

	       ( conj(re_kets(j) )   + im *       conj( im_kets(j) ) ) + //G(k)+
 	  
	         ( re_bras(j)   - im * im_bras(j)  ) * //G(k)
	         ( real( re_kets(j) ) + im * real( im_kets(j) ) );   //Re(G(k))
	

         //Here: w(k) += (dG(k)) * Re(G(k)) - Re(G(k)) * (dG(k))
         w[j] += (  re_D_bras(j)  - im *  im_D_bras(j)  )  * //dG(k)
		 ( real( re_kets(j)   ) + im * real( im_kets(j) ) ) - //Re(G(k))
	  
		 ( real( re_bras(j) )   - im * real( im_bras(j) ) )  *  //Re(G(k))
	         ( conj( re_D_kets(j) )         + im * conj ( im_D_kets(j) ) ); //dG(k)

     }
   }      
    
# pragma omp critical
   {
     for(int k = 0; k < nump_; k++){
       r_data[ k ]         += p[ k ];
       r_data[ k + nump_ ] += w[ k ];	
     }
   } 
 }
}




  
};
