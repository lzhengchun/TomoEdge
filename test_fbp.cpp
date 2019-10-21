#include <iostream>
#include <fstream>
#include <chrono>

#include "fbp.h"

#define PI (float)3.14159265358

using namespace std;

int main(int argc, char const *argv[])
{
    const unsigned int n_theta = 1500;
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
    float *recon_buf = new float[recon_buf_sz](); 
    float *sino_buf  = new float[sino_buf_sz]();

    std::ifstream sino_fin("sino-glass.bin", std::ios::binary);
    sino_fin.read((char *)sino_buf, sizeof(float) * sino_buf_sz);

    if(sino_fin){
        printf("sinogram loaded, %ld bytes\n", sino_fin.gcount());
    }else{
        printf("Error while load sinogram! EoF reached, only %ld bytes could be read\n", sino_fin.gcount());
        exit(-1);
    }

    auto e2e_st = chrono::steady_clock::now();
    fbp fbp_cu(theta, rot_center, n_theta, n_slice, col_size);
    auto comp_st = chrono::steady_clock::now();
    fbp_cu.linesum(recon_buf, sino_buf);
    auto comp_ed = chrono::steady_clock::now();

    printf("It takes %.3f ms to setup and %.3f ms to reconstruct!\n", \
           chrono::duration_cast<chrono::microseconds>(comp_st - e2e_st ).count()/1000.,\
           chrono::duration_cast<chrono::microseconds>(comp_ed - comp_st).count()/1000.);

    float checksum = 0;
    for (size_t i = 0; i < recon_buf_sz; i++){
        checksum += recon_buf[i];
    }
    printf("reconstruction checksum: %.3f\n", checksum);
    
    // dump output array to a file
    std::ofstream img_fout("output_img.bin", std::ios::out | std::ios::binary);
    img_fout.write((char *) recon_buf, sizeof(float) * recon_buf_sz );
    img_fout.close();

    return 0;
}

