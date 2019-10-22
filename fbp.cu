#include "kernels_fbp.cuh"
#include "fbp.h"

fbp::fbp(float *theta_, float center_, int Ntheta_, int Nz_, int N_)
{
	N = N_;
	Ntheta = Ntheta_;
	Nz = Nz_;
	center = center_;

	// arrays allocation on GPU
	cudaMalloc((void **)&f, N * N * Nz * sizeof(float));
	cudaMalloc((void **)&g, N * Ntheta * Nz * sizeof(float));
	cudaMalloc((void **)&fg, (N / 2 + 1) * Ntheta * Nz * sizeof(float2));
	cudaMalloc((void **)&theta, Ntheta * sizeof(float));

	//fft plans for filtering
	int ffts[] = {N};
	int idist = N;
	int odist = N / 2 + 1;
	int inembed[] = {N};
	int onembed[] = {N / 2 + 1};
	cufftPlanMany(&plan_forward, 1, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_R2C, Ntheta * Nz);
	cufftPlanMany(&plan_inverse, 1, ffts, onembed, 1, odist, inembed, 1, idist, CUFFT_C2R, Ntheta * Nz);

	//init thread blocks and block grids
	BS3d.x = 16;
	BS3d.y = 16;
	BS3d.z = 1;

	GS3d0.x = ceil(N / (float)BS3d.x);
	GS3d0.y = ceil(N / (float)BS3d.y);
	GS3d0.z = ceil(Nz / (float)BS3d.z);

	GS3d1.x = ceil(N / (float)BS3d.x);
	GS3d1.y = ceil(Ntheta / (float)BS3d.y);
	GS3d1.z = ceil(Nz / (float)BS3d.z);

	// copy angles to gpu
	cudaMemcpy(theta, theta_, Ntheta * sizeof(float), cudaMemcpyDefault);
}

fbp::~fbp()
{
	cudaFree(f);
	cudaFree(g);
	cudaFree(fg);
	cudaFree(theta);
	cufftDestroy(plan_forward);
	cufftDestroy(plan_inverse);
}

void fbp::linesum(float *f_, float *g_)
{
    size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);
    printf("%ld out of %ld bytes are free with FBP\n", mem_free, mem_total);
	// copy data, init arrays with 0
	cudaMemcpy(g, g_, N * Ntheta * Nz * sizeof(float), cudaMemcpyDefault);
	cudaMemset(f, 0, N * N * Nz * sizeof(float));
	//1d FFT
	cufftExecR2C(plan_forward, (cufftReal *)g, (cufftComplex *)fg);
	// filtering
	applyfilter<<<GS3d1, BS3d>>>(fg, N, Ntheta, Nz);
	// fft back
	cufftExecC2R(plan_inverse, (cufftComplex *)fg, (cufftReal *)g);
	// summation over lines
	linesum_ker<<<GS3d0, BS3d>>>(f, g, theta, center, N, Ntheta, Nz);
	// copy result to cpu
	cudaMemcpy(f_, f, N * N * Nz * sizeof(float), cudaMemcpyDefault);
}