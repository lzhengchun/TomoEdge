#include "fbp_cuda.h"
#include "kernels.cuh"
#include <stdio.h>
#include <iostream>

using namespace std;

fbp::fbp(float* theta_, float center_, int Ntheta_, int Nz_, int N_){
    N = N_;
    Ntheta = Ntheta_;
    Nz = Nz_;
    center = center_;

    // USFFT parameters
    float eps = 1e-3;
    mu = -log(eps) / (2 * N * N);
    M = ceil(2 * N * 1 / PI * sqrt(-mu * log(eps) + (mu * N) * (mu * N) / 4));

    // arrays allocation on GPU
    cudaErrchk( cudaMalloc((void **)&f, N * N * Nz * sizeof(float2)) );
    cudaErrchk( cudaMalloc((void **)&g, N * Ntheta * Nz * sizeof(float2)) );
    cudaErrchk( cudaMalloc((void **)&fde, 2 * N * 2 * N * Nz * sizeof(float2)) );
    cudaErrchk( cudaMalloc((void **)&fdee, (2 * N + 2 * M) * (2 * N + 2 * M) * Nz * sizeof(float2)) );
    cudaErrchk( cudaMalloc((void **)&x, N * Ntheta * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&y, N * Ntheta * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&theta, Ntheta * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&shiftadj, N * sizeof(float2)) );

    // init 2d FFTs
    int ffts[2] = {2 * N, 2 * N};
    int idist = 2 * N * 2 * N;
    int odist = (2 * N + 2 * M) * (2 * N + 2 * M);
    int inembed[2] = {2 * N, 2 * N};
    int onembed[2] = {2 * N + 2 * M, 2 * N + 2 * M};

    auto cufft_ret = cufftPlanMany(&plan2dadj, 2, ffts, onembed, 1, odist, inembed, 1, idist, CUFFT_C2C, Nz);
    if(cufft_ret != CUFFT_SUCCESS){
        printf("CUFFT error: cufftPlanMany failed, file: %s, line: %d\n", __FILE__, __LINE__);
        return;
    }

    // init 1d FFTs
    ffts[0] = N;
    idist = N;
    odist = N;
    inembed[0] = N;
    onembed[0] = N;
    cufft_ret = cufftPlanMany(&plan1d, 1, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, Ntheta * Nz);
    if(cufft_ret != CUFFT_SUCCESS){
        printf("CUFFT error: cufftPlanMany failed, file: %s, line: %d\n", __FILE__, __LINE__);
        return;
    }
    //init thread blocks and block grids
    BS3d.x = 16;
    BS3d.y = 16;
    GS2d0.x = ceil(N / (float)BS3d.x);
    GS2d0.y = ceil(Ntheta / (float)BS3d.y);

    GS3d0.x = ceil(N / (float)BS3d.x);
    GS3d0.y = ceil(N / (float)BS3d.y);
    GS3d0.z = ceil(Nz / (float)BS3d.z);

    GS3d1.x = ceil(2 * N / (float)BS3d.x);
    GS3d1.y = ceil(2 * N / (float)BS3d.y);
    GS3d1.z = ceil(Nz / (float)BS3d.z);

    GS3d2.x = ceil((2 * N + 2 * M) / (float)BS3d.x);
    GS3d2.y = ceil((2 * N + 2 * M) / (float)BS3d.y);
    GS3d2.z = ceil(Nz / (float)BS3d.z);
    
    GS3d3.x = ceil(N / (float)BS3d.x);
    GS3d3.y = ceil(Ntheta / (float)BS3d.y);
    GS3d3.z = ceil(Nz / (float)BS3d.z);

    // copy angles to gpu
    cudaErrchk( cudaMemcpy(theta, theta_, Ntheta * sizeof(float), cudaMemcpyHostToDevice));
    
    // compute polar coordinates
    takexy<<<GS2d0, BS3d>>>(x, y, theta, N, Ntheta);
    cudaErrchk(cudaPeekAtLastError());

    takeshift<<<ceil(N/1024.0), 1024>>>(shiftadj, (center - N / 2.0), N);
    cudaErrchk(cudaPeekAtLastError());
}

fbp::~fbp(){
    cudaFree(f);
    cudaFree(g);
    cudaFree(fde);
    cudaFree(fdee);
    cudaFree(x);
    cudaFree(y);
    cudaFree(shiftadj);
    cufftDestroy(plan2dadj);
    cufftDestroy(plan1d);
}

void fbp::adj(float2* f_, float2* g_){
    size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);
    printf("%ld out of %ld bytes are free\n", mem_free, mem_total);

    // copy data, init arrays with 0
    cudaErrchk( cudaMemcpy(g, g_, N * Ntheta * Nz * sizeof(float2), cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemset(fde,  0, 2 * N * 2 * N * Nz * sizeof(float2)) );
    cudaErrchk( cudaMemset(fdee, 0, (2 * N + 2 * M) * (2 * N + 2 * M) * Nz * sizeof(float2)) );
    cudaErrchk( cudaMemset(f,    0, N * N * Nz * sizeof(float2)) );

    // 1d FFT
    fftshift1c<<<GS3d3, BS3d>>>(g, N, Ntheta, Nz);
    cudaErrchk(cudaPeekAtLastError());

    auto ret = cufftExecC2C(plan1d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD);
    if(ret != CUFFT_SUCCESS){
        printf("CUFFT error: ExecC2C Forward failed\n");
        return;
    }

    fftshift1c<<<GS3d3, BS3d>>>(g, N, Ntheta, Nz);
    cudaErrchk(cudaPeekAtLastError());

    // shift with respect to given center
    shift<<<GS3d3, BS3d>>>(g, shiftadj, N, Ntheta, Nz);
    cudaErrchk(cudaPeekAtLastError());

    // filtering 
    applyfilter<<<GS3d3, BS3d>>>(g, N, Ntheta, Nz);
    cudaErrchk(cudaPeekAtLastError());

    // scattering from the polar grid
    scatter<<<GS3d3, BS3d>>>(fdee, g, x, y, M, mu, N, Ntheta, Nz);
    cudaErrchk(cudaPeekAtLastError());
    
    // wrap frequencies
    wrapadj<<<GS3d2, BS3d>>>(fdee, N, Nz, M);
    cudaErrchk(cudaPeekAtLastError());

    // 2d IFFT
    fftshiftc<<<GS3d2, BS3d>>>(fdee, 2 * N + 2 * M, Nz);
    cudaErrchk(cudaPeekAtLastError());

    ret = cufftExecC2C(plan2dadj, (cufftComplex *)&fdee[M + M * (2 * N + 2 * M)], (cufftComplex *)fde, CUFFT_INVERSE);
    if(ret != CUFFT_SUCCESS){
        printf("CUFFT error: ExecC2C inverse failed\n");
        return;
    }
    fftshiftc<<<GS3d1, BS3d>>>(fde, 2 * N, Nz);
    cudaErrchk(cudaPeekAtLastError());

    // divide by the USFFT kernel function with unpadding
    unpaddivphi<<<GS3d0, BS3d>>>(f, fde, mu, N, Nz);
    cudaErrchk(cudaPeekAtLastError());

    // copy result to cpu
    cudaErrchk( cudaMemcpy(f_, f, N * N * Nz * sizeof(float2), cudaMemcpyDeviceToHost) );
}
