#include <iostream>
#include <fstream>
#include <chrono>

#include "gridrec.h"
#include "argparse.hpp"

#define PI (float)3.14159265358

using namespace std;

int main(int argc, char const *argv[])
{
    int ss = 15;
    
    const unsigned int n_theta = 1500/ss;
    const float rot_center = 1427.;
    const unsigned int n_slice = 1;
    const unsigned int col_size = 2560;

    float* theta = new float[n_theta]();
    for (auto i = 0; i < n_theta; i++){
        theta[i] = i * PI / n_theta;
    }
    
    const unsigned int recon_buf_sz = col_size * col_size * n_slice;
    const unsigned int sino_buf_sz  = col_size * n_theta  * n_slice;

    // use unified memory address will help becuase CPU and GPU shares the same memory on TX2
    float2* recon_buf = new float2[recon_buf_sz](); 
    float2* sino_buf  = new float2[sino_buf_sz]();
    // cudaMallocHost((void **)&sino_buf, sino_buf_sz * sizeof(float2), cudaHostAllocDefault);

    std::ifstream sino_fin("sino-complex.bin", std::ios::binary);
    sino_fin.read((char *)sino_buf, sizeof(float2) * sino_buf_sz);

    if(sino_fin){
        printf("sinogram loaded, %ld bytes\n", sino_fin.gcount());
    }else{
        printf("Error while load sinogram! EoF reached, only %ld bytes could be read\n", sino_fin.gcount());
        exit(-1);
    }

    auto e2e_st = chrono::steady_clock::now();
    gridrec gridrec_cu(theta, rot_center, n_theta, n_slice, col_size);
    // warm up for benchmark test
    for (size_t i = 0; i < 0; i++){
        gridrec_cu.adj(recon_buf, sino_buf);
    }

    auto comp_st = chrono::steady_clock::now();
    gridrec_cu.adj(recon_buf, sino_buf);
    auto comp_ed = chrono::steady_clock::now();

    printf("It takes %.3f ms to setup and %.3f ms to reconstruct!\n", \
           chrono::duration_cast<chrono::microseconds>(comp_st - e2e_st ).count()/1000.,\
           chrono::duration_cast<chrono::microseconds>(comp_ed - comp_st).count()/1000.);

    float checksum_real = 0, checksum_imag = 0;
    for (size_t i = 0; i < recon_buf_sz; i++){
        checksum_real += recon_buf[i].x;
        checksum_imag += recon_buf[i].y;
    }
    printf("reconstruction checksum: %.3f + i%.3f\n", checksum_real, checksum_imag);
    
    // dump output array to a file
    std::ofstream img_fout("gridrec-output-img.bin", std::ios::out | std::ios::binary);
    img_fout.write((char *) recon_buf, sizeof(float2) * recon_buf_sz );
    img_fout.close();

    return 0;
}

