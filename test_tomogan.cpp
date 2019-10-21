#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda.h>

#include "tomoGAN.h"
// #include "utils.hpp"

int main(int argc, char const *argv[])
{
    const uint32 img_n = 4, img_c = 1, img_w = 1024, img_h = 1024;

    tomoGAN mdl;
    uint32 weight_sz = mdl.get_n_weights();
    float *weights = new float[weight_sz]();
    // float *weights;
    // cudaHostAlloc((void**)&weights, weight_sz * sizeof(float), cudaHostAllocPortable);
    std::ifstream weights_fin("tomogan_weights_serilize-oihw-1ch.bin", std::ios::binary);
    weights_fin.read((char *)weights, sizeof(float) * weight_sz);
    if(weights_fin){
        printf("%ld bytes of weights have been successfully read\n", weights_fin.gcount());
    }else{
        printf("Error while load input, EoF reached, only %ld bytes could be read\n", weights_fin.gcount());
        exit(-1);
    }
    weights_fin.close();

    mdl.model_init(img_n, img_c, img_h, img_w, weights);

    const uint32 input_size = img_n * img_c * img_h * img_w;
    const uint32 output_size = img_n * 1 * img_h * img_w;
    float *input_buf  = new float[input_size]();
    float *output_buf = new float[output_size]();
    // float *input_buf, *output_buf;
    // cudaHostAlloc((void**)&input_buf, input_size * sizeof(float), cudaHostAllocPortable);
    // cudaHostAlloc((void**)&output_buf, output_size * sizeof(float), cudaHostAllocPortable);

    std::ifstream inputs_fin("test_input_serilize-nchw.bin", std::ios::binary);
    inputs_fin.read((char *)input_buf, sizeof(float) * input_size);
    if(inputs_fin){
        printf("%ld bytes of input data have been successfully read\n", inputs_fin.gcount());
    }else{
        printf("Error while load input, EoF reached, only %ld bytes could be read\n", inputs_fin.gcount());
        exit(-1);
    }
    inputs_fin.close();

    for (size_t i = 0; i < 10; i++){
        mdl.predict(input_buf, output_buf);
    }

    double checksum = 0;
    for (auto i = 0; i < output_size; i++){
        checksum += output_buf[i];
    }
    printf("output checksum: %.3lf\n", checksum);

    // dump output array to a file
    std::ofstream img_fout("output_img.bin", std::ios::out | std::ios::binary);
    img_fout.write((char *) output_buf, sizeof(float) * output_size);
    img_fout.close();

    // free memory allocations
    delete[] weights;
    delete[] input_buf;
    delete[] output_buf;

    // cudaFreeHost(weights);
    // cudaFreeHost(input_buf);
    // cudaFreeHost(output_buf);
    return 0;
}
