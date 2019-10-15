#include "tomoGAN.h"

tomoGAN::tomoGAN(uint32 img_n, uint32 img_c, uint32 img_h, uint32 img_w, float *weights_h){
    const uint32 box1_o_sz_h = img_h;
    const uint32 box1_o_sz_w = img_w;
    const uint32 box2_o_sz_h = img_h/2;
    const uint32 box2_o_sz_w = img_w/2;
    const uint32 box3_o_sz_h = img_h/4;
    const uint32 box3_o_sz_w = img_w/4;
    const uint32 intr_o_sz_h = img_h/8;
    const uint32 intr_o_sz_w = img_w/8;

    cudaErrchk( cudaMalloc((void **)&input_buf,  img_n * img_c * img_h * img_w * sizeof(uint8)) );
    cudaErrchk( cudaMalloc((void **)&output_buf, img_n * 1     * img_h * img_w * sizeof(uint8)) );
    cudaErrchk( cudaMalloc((void **)&layer_buf1, img_n * 32    * img_h * img_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&layer_buf2, img_n * 64    * img_h * img_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&box1_out_buf, img_n * 32  * box1_o_sz_h * box1_o_sz_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&box2_out_buf, img_n * 64  * box2_o_sz_h * box2_o_sz_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&box3_out_buf, img_n * 128 * box3_o_sz_h * box3_o_sz_w * sizeof(float)) );

    uint32 w_acc_sz = 0;
    for(auto i = 0; i <= 18; i++){
        auto w_sz = conv_sz[i] * conv_sz[i] * conv_ch[i] * n_conv[i];
        cudaErrchk( cudaMalloc((void **)&(weights[i]), w_sz * sizeof(float)) );
        cudaErrchk( cudaMemcpy(weights[i], weights + w_acc_sz, w_sz * sizeof(float), cudaMemcpyHostToDevice) );
        w_acc_sz += w_sz;
    }

}

tomoGAN::~tomoGAN(){
    cudaFree(input_buf);
    cudaFree(output_buf);
    cudaFree(layer_buf1);
    cudaFree(layer_buf2);
    cudaFree(box1_out_buf);
    cudaFree(box2_out_buf);
    cudaFree(box3_out_buf);
}

void tomoGAN::inference(uint8 *img_in, uint8 *img_out){
    
}