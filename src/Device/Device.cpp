#include"Device.hpp"
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/SparseCore>
#include<chrono>
#include<iostream>
#include<fstream>


void Device::Anderson_disorder(r_type disorder_vec[]){

  size_t SUBDIM = device_vars_.SUBDIM_;
  r_type str = device_vars_.dis_str_;
  
  
  for(size_t i=0;i<SUBDIM; i++){
    r_type random_potential = str * this->rng().get()-str/2;

    
    disorder_vec[i] = random_potential;
  }
  
}




void Device::Anderson_disorder(){

  size_t SUBDIM = device_vars_.SUBDIM_;
  r_type str = device_vars_.dis_str_;
  

        

  for(size_t i=0;i<SUBDIM; i++){
    r_type random_potential = str * this->rng().get()-str/2;
    
    dis_[i] = random_potential;
  }
  
}



void Device::rearrange_initial_vec(type r_vec[]){ //supe duper hacky; Standard for 2-terminal devices.
  size_t Dim = this->parameters().DIM_,
    subDim = this->parameters().SUBDIM_;

  size_t C   = this->parameters().C_,
      Le  = this->parameters().LE_,
      W   = this->parameters().W_;

  type tmp[subDim];

#pragma omp parallel for
    for(size_t n=0;n<subDim;n++)
      tmp[n]=r_vec[n];

#pragma omp parallel for
    for(size_t n=0;n<Dim;n++)
      r_vec[n] = 0;
        

#pragma omp parallel for
    for(size_t n=0;n<Le*W;n++)
      r_vec[C*W + n ]=tmp[ n];

}

void Device::traceover(type* traced, type* full_vec, int s, int num_reps){ //standard for 2-terminal devices.
  size_t subDim = this->parameters().SUBDIM_,
      C   = this->parameters().C_,
      W   = this->parameters().W_,
      sec_size = subDim/num_reps,
      buffer_length = sec_size;
	
  if( s == num_reps-1 )
      buffer_length += subDim % num_reps;

      
#pragma omp parallel for 
      for(size_t i=0;i<buffer_length;i++)
        traced[i] = full_vec[s*sec_size + i+C*W];

  };


