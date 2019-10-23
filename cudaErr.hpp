#ifndef CUDA_ERR_HPP
#define CUDA_ERR_HPP

#include <cuda.h>
#include <cudnn.h>
#include <string>
#include <iostream>
#include <chrono>

//function to print out error message from cuDNN calls
#define cudnnErrchk(exp){ \
    cudnnStatus_t status = (exp); \
    if(status != CUDNN_STATUS_SUCCESS) { \
      std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } 

#define cudaErrchk(ans)  cudaAssert((ans), __FILE__, __LINE__) 
inline void cudaAssert(cudaError_t code, std::string file, int line){
    if (code != cudaSuccess){
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << "; file: " << file << ", line:" << line << std::endl;
        exit(-1);
    }
}

#endif