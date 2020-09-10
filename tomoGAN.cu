#include "tomoGAN.h"

void __global__ kernel_upsampling(float *input, uint32 n, uint32 chs, uint32 height, uint32 width, float *output){
    int t_col = blockDim.x * blockIdx.x + threadIdx.x;
    int t_row = blockDim.y * blockIdx.y + threadIdx.y;
    int t_ch  = blockDim.z * blockIdx.z + threadIdx.z;
    if(t_row >= height || t_col >= width || t_ch >= chs){
        return;
    }

    unsigned int urow = 2 * t_row;
    unsigned int ucol = 2 * t_col;
    unsigned int uwidth = 2 * width;
    unsigned int uheight= 2 * height;

    unsigned int n_offset = height * width * chs;
    for(uint i = 0; i < n; i++){
        float pixel = input[i * n_offset + height * width * t_ch + width * t_row + t_col];
        output[i * 4 * n_offset + uheight * uwidth  * t_ch + uwidth * urow + ucol] = pixel;  // [ch][urow][ucol]
        output[i * 4 * n_offset + uheight * uwidth  * t_ch + uwidth * (urow+1) + ucol] = pixel;  // [ch][urow+1][ucol] 
        output[i * 4 * n_offset + uheight * uwidth  * t_ch + uwidth * urow + (ucol+1)] = pixel;  // [ch][urow][ucol+1]
        output[i * 4 * n_offset + uheight * uwidth  * t_ch + uwidth * (urow+1) + (ucol+1)] = pixel;  // [ch][urow+1][ucol+1]
    }
}

uint32 tomoGAN::get_n_weights(){
    uint32 layers = sizeof(conv_ch) / sizeof(uint32);
    uint32 weight_sz = 0;
    for (auto l = 0; l < layers; l++){
        weight_sz += conv_ch[l] * conv_sz[l] * conv_sz[l] * n_conv[l] + n_conv[l];
    }
    return weight_sz;
}

void tomoGAN::model_init(uint32 img_n, uint32 img_c, uint32 img_h, uint32 img_w, float *weights_h){
    cudnnErrchk(cudnnCreate(&cudnn_handle));
    n_img_in = img_n;
    c_img_in = img_c;
    h_img_in = img_w;
    w_img_in = img_w;
    const uint32 box1_o_sz_h = img_h;
    const uint32 box1_o_sz_w = img_w;
    const uint32 box2_o_sz_h = img_h/2;
    const uint32 box2_o_sz_w = img_w/2;
    const uint32 box3_o_sz_h = img_h/4;
    const uint32 box3_o_sz_w = img_w/4;
    
    cudaErrchk( cudaMalloc((void **)&input_buf,  img_n * img_c * img_h * img_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&output_buf, img_n * 1     * img_h * img_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&layer_buf1, img_n * 64    * img_h * img_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&layer_buf2, img_n * 32    * img_h * img_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&box1_out_buf, img_n * 32  * box1_o_sz_h * box1_o_sz_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&box2_out_buf, img_n * 64  * box2_o_sz_h * box2_o_sz_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&box3_out_buf, img_n * 128 * box3_o_sz_h * box3_o_sz_w * sizeof(float)) );
    
    uint32 w_acc_sz = 0;
    for(auto i = 0; i <= 15; i++){
        auto w_sz = conv_sz[i] * conv_sz[i] * conv_ch[i] * n_conv[i] + n_conv[i];
        cudaErrchk( cudaMalloc((void **)&(weights_d[i]), w_sz * sizeof(float)) );
        cudaErrchk( cudaMemcpy(weights_d[i], weights_h + w_acc_sz, w_sz * sizeof(float), cudaMemcpyHostToDevice) );
        w_acc_sz += w_sz;
    }
    printf("%d weigths has been loaded and copied to device memory\n", w_acc_sz);

    cudnnErrchk( cudnnCreateTensorDescriptor(&bias_desc) );
    
    cudnnErrchk( cudnnCreateActivationDescriptor(&relu_activ_desc) );
    cudnnErrchk( cudnnSetActivationDescriptor(relu_activ_desc,
                                              CUDNN_ACTIVATION_RELU,
                                              CUDNN_PROPAGATE_NAN,
                                              0.0) );        
}

tomoGAN::~tomoGAN(){
    cudaFree(input_buf);
    cudaFree(output_buf);
    cudaFree(layer_buf1);
    cudaFree(layer_buf2);
    cudaFree(box1_out_buf);
    cudaFree(box2_out_buf);
    cudaFree(box3_out_buf);

    cudnnErrchk( cudnnDestroyTensorDescriptor(bias_desc) );
    cudnnErrchk( cudnnDestroyActivationDescriptor(relu_activ_desc) );
}

void tomoGAN::upsampling(float *mem_in, float *mem_out,
                    cudnnTensorDescriptor_t &tensor_in_desc, \
                    cudnnTensorDescriptor_t &tensor_out_desc){

    int n, c, h, w, ns, cs, hs, ws;
    cudnnDataType_t dt;
    cudnnErrchk(cudnnGetTensor4dDescriptor(tensor_in_desc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws));

    cudnnErrchk(cudnnSetTensor4dDescriptor(tensor_out_desc, 
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           n, c, 2*h, 2*w)); 

    dim3 gs, bs(8, 8, 8);
    gs.x = ceil((float)w / bs.x);
    gs.y = ceil((float)h / bs.y);
    gs.z = ceil((float)c / bs.z);
    kernel_upsampling<<<gs, bs>>>(mem_in, n, c, h, w, mem_out);    
    cudaDeviceSynchronize();
}

void tomoGAN::maxpooling(float *mem_in, float *mem_out,
                        cudnnTensorDescriptor_t &tensor_in_desc, \
                        cudnnTensorDescriptor_t &tensor_out_desc){

    cudnnPoolingDescriptor_t pooling_desc;
    cudnnErrchk(cudnnCreatePoolingDescriptor(&pooling_desc));
    //initialize descriptor
    cudnnErrchk(cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                            CUDNN_POOLING_MAX,       //mode - max pooling
                                            CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                            2,                       //window height
                                            2,                       //window width
                                            0,                       //vertical padding
                                            0,                       //horizontal padding
                                            2,                       //vertical stride
                                            2));                     //horizontal stride

    int out_n, out_c, out_h, out_w;
    cudnnErrchk(cudnnGetPooling2dForwardOutputDim(pooling_desc, \
                                                  tensor_in_desc, \
                                                  &out_n, &out_c, &out_h, &out_w));

    cudnnErrchk(cudnnSetTensor4dDescriptor(tensor_out_desc,          //descriptor handle
                                           CUDNN_TENSOR_NCHW,        //data format
                                           CUDNN_DATA_FLOAT,         //data type (precision)
                                           out_n, out_c, out_h, out_w));   

    float alpha = 1, beta = 0;
    cudnnErrchk(cudnnPoolingForward(cudnn_handle,    //cuDNN context handle
                                    pooling_desc,    //pooling descriptor handle
                                    &alpha,          //alpha scaling factor
                                    tensor_in_desc,  //input tensor descriptor
                                    mem_in,          //input data pointer to GPU memory
                                    &beta,           //beta scaling factor
                                    tensor_out_desc, //output tensor descriptor
                                    mem_out));       //output data pointer from GPU memory      
}

void tomoGAN::conv2d(uint32 in_ch, uint8 knl_sz, uint32 n_knl, \
                     float *mem_in, float *mem_out, float *weights,\
                     cudnnTensorDescriptor_t &tensor_in_desc, \
                     cudnnTensorDescriptor_t &tensor_out_desc, 
                     bool relu){
    cudnnFilterDescriptor_t kernel_desc;
    cudnnErrchk(cudnnCreateFilterDescriptor(&kernel_desc));
    cudnnErrchk(cudnnSetFilter4dDescriptor(kernel_desc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,/*filter layout will be KCRS, i.e., OIHW*/
                                           n_knl, 
                                           in_ch,
                                           knl_sz, knl_sz));

    cudnnConvolutionDescriptor_t conv2d_desc;
    cudnnErrchk(cudnnCreateConvolutionDescriptor(&conv2d_desc));
    cudnnErrchk(cudnnSetConvolution2dDescriptor(conv2d_desc,
                                                knl_sz/2,/*pad_height=*/
                                                knl_sz/2,/*pad_width=*/
                                                1,/*vertical_stride=*/
                                                1,/*horizontal_stride=*/
                                                1,/*dilation_height=*/
                                                1,/*dilation_width=*/
                                                CUDNN_CROSS_CORRELATION,/*mode=*/
                                                CUDNN_DATA_FLOAT));/*computeType=*/
    int out_n, out_c, out_h, out_w; // nchw
    cudnnErrchk(cudnnGetConvolution2dForwardOutputDim(conv2d_desc, \
                                                      tensor_in_desc, \
                                                      kernel_desc, \
                                                      &out_n, &out_c, &out_h, &out_w));

    cudnnErrchk(cudnnSetTensor4dDescriptor(tensor_out_desc,
                                           CUDNN_TENSOR_NCHW,/*format=*/
                                           CUDNN_DATA_FLOAT, /*dataType=*/
                                           out_n, out_c, out_h, out_w)); /*image_width=*/

    // int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    // int returnedAlgoCount = -1;
    // cudnnConvolutionFwdAlgoPerf_t algo_perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    // cudnnErrchk(cudnnFindConvolutionForwardAlgorithm(cudnn_handle,
    //                                                  tensor_in_desc,
    //                                                  kernel_desc,
    //                                                  conv2d_desc,
    //                                                  tensor_out_desc,
    //                                                  requestedAlgoCount,
    //                                                  &returnedAlgoCount,
    //                                                  algo_perf));
    // printf("requestedAlgoCount: %d, returnedAlgoCount: %d, fastest: %d\n", \
    //        requestedAlgoCount, returnedAlgoCount, algo_perf[0].algo);

    cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    // cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    size_t workspace_bytes = 0;
    cudnnErrchk(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                        tensor_in_desc,
                                                        kernel_desc,
                                                        conv2d_desc,
                                                        tensor_out_desc,
                                                        algo,
                                                        &workspace_bytes));
    // std::cout << "Workspace size: " << (workspace_bytes / 1024.) << " KiB" << std::endl;

    void* d_workspace{nullptr};
    if(workspace_bytes > 0){
        cudaErrchk( cudaMalloc(&d_workspace, workspace_bytes) );
    }

    float alpha = 1, beta = 0;
    cudnnErrchk(cudnnConvolutionForward(cudnn_handle,
                                        &alpha,
                                        tensor_in_desc, mem_in,
                                        kernel_desc, weights,
                                        conv2d_desc, algo,
                                        d_workspace, workspace_bytes,
                                        &beta,
                                        tensor_out_desc, mem_out));
    cudaFree(d_workspace);

    // add bias
    alpha = 1, beta = 1;
    cudnnErrchk(cudnnSetTensor4dDescriptor(bias_desc, 
                                           CUDNN_TENSOR_NCHW, 
                                           CUDNN_DATA_FLOAT, 
                                           1, n_knl, 1, 1));

    cudnnErrchk(cudnnAddTensor(cudnn_handle,
                               &alpha, 
                               bias_desc, weights + n_knl * knl_sz * knl_sz * in_ch,
                               &beta,
                               tensor_out_desc, mem_out));

    if(relu){
        alpha = 1, beta = 0;
        cudnnErrchk(cudnnActivationForward(cudnn_handle,
                                           relu_activ_desc,
                                           &alpha,
                                           tensor_out_desc, mem_out,
                                           &beta,
                                           tensor_out_desc, mem_out));
    }

}

void tomoGAN::prinf_tensor_sz(cudnnTensorDescriptor_t tensor){
    int n, c, h, w, ns, cs, hs, ws;
    cudnnDataType_t dt;
    cudnnErrchk(cudnnGetTensor4dDescriptor(tensor, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws));
    printf("Tensor SZ: %3d x %3d x %3d x %3d \n", n, c, h, w);
}

void tomoGAN::concat(float *box_out, float *conv_out, float *dst_mem, 
                     uint32 batch_sz, uint32 c, uint32 h, uint32 w,
                     cudnnTensorDescriptor_t& out_tensor_desc){
    uint32 batch_offset = c * h * w;
    for(auto n = 0; n < batch_sz; n++){
        cudaErrchk( cudaMemcpy(dst_mem + 2 * n * batch_offset, 
                               box_out + n * batch_offset, 
                               batch_offset * sizeof(float), cudaMemcpyDeviceToDevice) );

        cudaErrchk( cudaMemcpy(dst_mem + (2 * n + 1) * batch_offset, 
                               conv_out + n * batch_offset, 
                               batch_offset * sizeof(float), cudaMemcpyDeviceToDevice) );

        cudnnErrchk(cudnnSetTensor4dDescriptor(out_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_sz, c * 2, h, w)); 
    }
}

void tomoGAN::printf_layer_io(cudnnTensorDescriptor_t tensor_in, cudnnTensorDescriptor_t tensor_out){
    int n, c, h, w, ns, cs, hs, ws;
    cudnnDataType_t dt;
    cudnnErrchk(cudnnGetTensor4dDescriptor(tensor_in, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws));
    printf("NCHW: %3d x %3d x %4d x %4d => ", n, c, h, w);

    cudnnErrchk(cudnnGetTensor4dDescriptor(tensor_out, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws));
    printf("%3d x %3d x %4d x %4d \n", n, c, h, w);
}

void tomoGAN::predict(float *img_in, float *img_out){
    cudnnTensorDescriptor_t tmp_tensor_desc1, tmp_tensor_desc2;
    cudnnErrchk(cudnnCreateTensorDescriptor(&tmp_tensor_desc1));
    cudnnErrchk(cudnnSetTensor4dDescriptor(tmp_tensor_desc1, 
                                           CUDNN_TENSOR_NCHW, 
                                           CUDNN_DATA_FLOAT, 
                                           n_img_in, c_img_in, h_img_in, w_img_in));

    cudnnErrchk(cudnnCreateTensorDescriptor(&tmp_tensor_desc2));

    cudaErrchk( cudaMemcpy(input_buf, img_in, \
                           n_img_in * c_img_in * h_img_in * w_img_in * sizeof(float), \
                           cudaMemcpyHostToDevice) );

    auto predict_st = chrono::steady_clock::now();

    conv2d(conv_ch[0], conv_sz[0], n_conv[0],  input_buf, layer_buf1,   weights_d[0], tmp_tensor_desc1, tmp_tensor_desc2, true);
    // printf_layer_io(tmp_tensor_desc1, tmp_tensor_desc2);

    // downsampling box 1
    conv2d(conv_ch[1], conv_sz[1], n_conv[1], layer_buf1, layer_buf2,   weights_d[1], tmp_tensor_desc2, tmp_tensor_desc1, true);
    conv2d(conv_ch[2], conv_sz[2], n_conv[2], layer_buf2, box1_out_buf, weights_d[2], tmp_tensor_desc1, tmp_tensor_desc2, true);
    maxpooling(box1_out_buf, layer_buf1, tmp_tensor_desc2, tmp_tensor_desc1);

    // downsampling box 2
    conv2d(conv_ch[3], conv_sz[3], n_conv[3], layer_buf1, layer_buf2,   weights_d[3], tmp_tensor_desc1, tmp_tensor_desc2, true);
    conv2d(conv_ch[4], conv_sz[4], n_conv[4], layer_buf2, box2_out_buf, weights_d[4], tmp_tensor_desc2, tmp_tensor_desc1, true);
    maxpooling(box2_out_buf, layer_buf1, tmp_tensor_desc1, tmp_tensor_desc2);

    // downsampling box 3
    conv2d(conv_ch[5], conv_sz[5], n_conv[5], layer_buf1, layer_buf2,   weights_d[5], tmp_tensor_desc2, tmp_tensor_desc1, true);
    conv2d(conv_ch[6], conv_sz[6], n_conv[6], layer_buf2, box3_out_buf, weights_d[6], tmp_tensor_desc1, tmp_tensor_desc2, true);
    maxpooling(box3_out_buf, layer_buf1, tmp_tensor_desc2, tmp_tensor_desc1);

    // intermediate 
    conv2d(conv_ch[7], conv_sz[7], n_conv[7], layer_buf1, layer_buf2, weights_d[7], tmp_tensor_desc1, tmp_tensor_desc2, true);

    upsampling(layer_buf2, layer_buf1, tmp_tensor_desc2, tmp_tensor_desc1);
    
    // upsampling box 1
    concat(box3_out_buf, layer_buf1, layer_buf2, n_img_in, 128, h_img_in / 4, w_img_in / 4, tmp_tensor_desc2);
    conv2d(conv_ch[8], conv_sz[8], n_conv[8], layer_buf2, layer_buf1, weights_d[8], tmp_tensor_desc2, tmp_tensor_desc1, true);
    conv2d(conv_ch[9], conv_sz[9], n_conv[9], layer_buf1, layer_buf2, weights_d[9], tmp_tensor_desc1, tmp_tensor_desc2, true);
    
    upsampling(layer_buf2, layer_buf1, tmp_tensor_desc2, tmp_tensor_desc1);

    // upsampling box 2
    concat(box2_out_buf, layer_buf1, layer_buf2, n_img_in, 64, h_img_in / 2, w_img_in / 2, tmp_tensor_desc1);
    conv2d(conv_ch[10], conv_sz[10], n_conv[10], layer_buf2, layer_buf1, weights_d[10], tmp_tensor_desc1, tmp_tensor_desc2, true);
    conv2d(conv_ch[11], conv_sz[11], n_conv[11], layer_buf1, layer_buf2, weights_d[11], tmp_tensor_desc2, tmp_tensor_desc1, true);

    upsampling(layer_buf2, layer_buf1, tmp_tensor_desc1, tmp_tensor_desc2);

    // upsampling box 3
    concat(box1_out_buf, layer_buf1, layer_buf2, n_img_in, 32, h_img_in, w_img_in, tmp_tensor_desc1);
    conv2d(conv_ch[12], conv_sz[12], n_conv[12], layer_buf2, layer_buf1, weights_d[12], tmp_tensor_desc1, tmp_tensor_desc2, true);
    conv2d(conv_ch[13], conv_sz[13], n_conv[13], layer_buf1, layer_buf2, weights_d[13], tmp_tensor_desc2, tmp_tensor_desc1, true);
    
    // output layers
    conv2d(conv_ch[14], conv_sz[14], n_conv[14], layer_buf2, layer_buf1, weights_d[14], tmp_tensor_desc1, tmp_tensor_desc2, true);
    conv2d(conv_ch[15], conv_sz[15], n_conv[15], layer_buf1, layer_buf2, weights_d[15], tmp_tensor_desc2, tmp_tensor_desc1, false);

    auto predict_ed = chrono::steady_clock::now();
    size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);

    printf("It takes %.3f ms to predict (inference), %ld out of %ld bytes are free\n", \
           chrono::duration_cast<chrono::microseconds>(predict_ed - predict_st ).count()/1000., mem_free, mem_total);

    cudaErrchk( cudaMemcpy(img_out, layer_buf2, n_img_in * 1 * h_img_in * w_img_in * sizeof(float), cudaMemcpyDeviceToHost) );
}