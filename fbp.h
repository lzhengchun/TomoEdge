#include <cuda.h>
#include <cudnn.h>
#include <cufft.h>
#include <string>
#include <iostream>

using namespace std;

#define cudaErrchk(ans)  cudaAssert((ans), __FILE__, __LINE__) 
inline void cudaAssert(cudaError_t code, string file, int line){
    if (code != cudaSuccess){
        cerr << "CUDA Error: " << cudaGetErrorString(code) << "; file: " << file << ", line:" << line << endl;
        exit(-1);
    }
}

class fbp
{
    int N;
    int Ntheta;
    int Nz;
    float center;

    float *f;
    float *g;
    float2 *fg;
    float *theta;

    cufftHandle plan_forward;
    cufftHandle plan_inverse;

    dim3 BS3d;
    dim3 GS3d0;
    dim3 GS3d1;

public:
    fbp(float *theta, float center_, int Ntheta, int Nz, int N);
    ~fbp();
    void linesum(float *f, float *g);
};
