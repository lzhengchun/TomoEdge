#include <cuda.h>
#include <cudnn.h>
#include <cufft.h>
#include <string>
#include <iostream>

#include "cudaErr.hpp"

using namespace std;

// #define cudaErrchk(ans)  cudaAssert((ans), __FILE__, __LINE__) 
// inline void cudaAssert(cudaError_t code, string file, int line){
//     if (code != cudaSuccess){
//         cerr << "CUDA Error: " << cudaGetErrorString(code) << "; file: " << file << ", line:" << line << endl;
//         exit(-1);
//     }
// }

class gridrec
{
    int N;
    int Ntheta;
    int Nz;
    int M;
    float mu;
    float center;

    float2 *f, *g, *ff, *gg, *f0, *g0;
    float *theta;

    float *x, *y;
    float2 *shiftadj;

    float2 *fde, *fdee;

    cufftHandle plan2dadj;
    cufftHandle plan1d;

    dim3 BS3d;
    dim3 GS2d0;
    dim3 GS3d0;
    dim3 GS3d1;
    dim3 GS3d2;
    dim3 GS3d3;

public:
    gridrec(float* theta, float center_, int Ntheta, int Nz, int N);
    ~gridrec();
    void adj(float2* f, float2* g);
};
