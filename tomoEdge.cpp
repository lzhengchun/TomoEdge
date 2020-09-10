#include <iostream>
#include <fstream>
#include <chrono>

#include "gridrec.h"
#include "tomoGAN.h"
#include "utils.hpp"

#define PI (float)3.14159265358

using namespace std;

int main(int argc, char const *argv[])
{
    // init Tomo
    int ss = 1;
    
    const unsigned int n_theta = 1500/ss;
    const float rot_center = 1427;// * (1024. / 2560.);
    const unsigned int n_slice = 1;
    const unsigned int col_size = 2560;

    float* theta = new float[n_theta]();
    for (auto i = 0; i < n_theta; i++){
        theta[i] = i * PI / n_theta;
    }
    gridrec gridrec_cu(theta, rot_center, n_theta, n_slice, col_size);

    // init TomoGAN model
    const uint32 img_n = 2, img_c = 1, img_w = 2560, img_h = 2560;

    tomoGAN mdl;
    uint32 weight_sz = mdl.get_n_weights();
    float *weights = new float[weight_sz]();

    std::ifstream weights_fin("tomogan_weights_serilize-oihw-1ch.bin", std::ios::binary);
    weights_fin.read((char *)weights, sizeof(float) * weight_sz);
    if(weights_fin){
        printf("%ld bytes of weights have been successfully read\n", weights_fin.gcount());
    }else{
        printf("Error while load input, EoF reached, only %ld bytes could be read\n", weights_fin.gcount());
        exit(-1);
    }
    weights_fin.close();

    mdl.model_init(img_n/2, img_c, img_h, img_w, weights);

    const uint32 tomogan_input_size = img_n * img_c * img_h * img_w;
    const uint32 tomogan_output_size = img_n * 1 * img_h * img_w;
    float *tomogan_input_buf  = new float[tomogan_input_size]();
    float *tomogan_output_buf = new float[tomogan_output_size]();

    // start tomo and denoise
    const unsigned int recon_buf_sz = col_size * col_size * n_slice;
    const unsigned int sino_buf_sz  = col_size * n_theta  * n_slice;

    float2* recon_buf = new float2[recon_buf_sz](); 
    float2* sino_buf  = new float2[sino_buf_sz]();

    std::ifstream sino_fin("sino-complex.bin", std::ios::binary);
    sino_fin.read((char *)sino_buf, sizeof(float2) * sino_buf_sz);

    if(sino_fin){
        printf("sinogram loaded, %ld bytes\n", sino_fin.gcount());
    }else{
        printf("Error while load sinogram! EoF reached, only %ld bytes could be read\n", sino_fin.gcount());
        exit(-1);
    }
    
    for (size_t rp = 0; rp < 1; rp++){    
        auto e2e_st = chrono::steady_clock::now();
        // start tomo
        gridrec_cu.adj(recon_buf, sino_buf);
        auto tomo_ed = chrono::steady_clock::now();

        // normalize and reorder to nchw
        minmax2uint8_reorder((float*)recon_buf, tomogan_input_buf, tomogan_input_size);
        auto norm_ed = chrono::steady_clock::now();

        // denoise
        mdl.predict(tomogan_input_buf, tomogan_output_buf);
        mdl.predict(tomogan_input_buf+tomogan_input_size/2, tomogan_output_buf+tomogan_input_size/2);
        auto dn_ed = chrono::steady_clock::now();

        float tomo_elaps = chrono::duration_cast<chrono::microseconds>(tomo_ed - e2e_st  ).count()/1000.;
        float norm_elaps = chrono::duration_cast<chrono::microseconds>(norm_ed - tomo_ed ).count()/1000.;
        float dn_elapse  = chrono::duration_cast<chrono::microseconds>(dn_ed   - norm_ed ).count()/1000.;

        printf("TOMO: %.2fms, Norm: %.2fms, TomoGAN: %.2f!\n", tomo_elaps, norm_elaps, dn_elapse);
    }

    // dump output array to a file
    std::ofstream img_fout("output_img.bin", std::ios::out | std::ios::binary);
    img_fout.write((char *) tomogan_output_buf, sizeof(float) * tomogan_output_size);
    img_fout.close();

    return 0;
}

