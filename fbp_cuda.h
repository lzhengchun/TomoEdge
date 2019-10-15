#include <cufft.h>
#include <iostream>

using namespace std;



class fbp
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
    fbp(float* theta, float center_, int Ntheta, int Nz, int N);
    ~fbp();
    void adj(float2* f, float2* g);
};
