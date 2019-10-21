#define PI 3.141592653589793238

void __global__ linesum_ker(float *f, float *g, float *theta, float center, int N, int Ntheta, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx >= N || ty >= N || tz >= Nz)
		return;
	float sp = 0;
	float f0 = 0;
	int s0 = 0;
	int ind = 0;
	for (int k = 0; k < Ntheta; k++)
	{
		sp = (tx - N / 2) * __cosf(theta[k]) - (ty - N / 2) * __sinf(theta[k]) + center; //polar coordinate
		//linear interpolation
		s0 = roundf(sp);
		ind = k * N * Nz + tz * N + s0;
		if ((s0 >= 0) & (s0 < N - 1))
			f0 += g[ind] + (g[ind + 1] - g[ind]) * (sp - s0) / N; 
	}
	f[tx + ty * N + tz * N * N] = f0;
}

void __global__ applyfilter(float2 *f, int N, int Ntheta, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx >= N / 2 + 1 || ty >= Ntheta || tz >= Nz)
		return;
	int id0 = tx + ty * (N / 2 + 1) + tz * Ntheta * (N / 2 + 1);
	float rho = tx / (float)N;
	float w = 0;
	if (rho != 0)
	{
		float c = (1 - fabs(rho) * 2);
		w = fabs(rho) * 4 * c * c * c; //Parzen
	}
	//add normalization constant for data
	w /= (Ntheta * sqrtf(PI / 2) * N);

	f[id0].x *= w;
	f[id0].y *= w;
}