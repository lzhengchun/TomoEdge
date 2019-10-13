#include <cufft.h>

class fbp
{
    size_t N;
    size_t Ntheta;
    size_t Nz;
    size_t M;
    float mu;
    float center;

    float2 *f;
    float2 *g;
    float2 *ff;
    float2 *gg;
    float2 *f0;
    float2 *g0;
    float *theta;

    float *x;
    float *y;
    float2 *shiftfwd;
    float2 *shiftadj;

    float2 *fde;
    float2 *fdee;

    cufftHandle plan2dfwd;
    cufftHandle plan2dadj;
    cufftHandle plan1d;

    dim3 BS3d;
    dim3 GS2d0;
    dim3 GS3d0;
    dim3 GS3d1;
    dim3 GS3d2;
    dim3 GS3d3;

public:
    fbp(float* theta, float center_, unsigned int Ntheta, unsigned int Nz, unsigned int N);
    ~fbp();
    void fwd(float2* g, float2* f);
    void adj(float2* f, float2* g);
};
