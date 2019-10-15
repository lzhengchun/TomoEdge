#include "tomoGAN.h"

tomoGAN::tomoGAN(uint32 img_n, uint32 img_c, uint32 img_h, uint32 img_w, float *weights_h){
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
    cudaErrchk( cudaMalloc((void **)&layer_buf1, img_n * 32    * img_h * img_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&layer_buf2, img_n * 64    * img_h * img_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&box1_out_buf, img_n * 32  * box1_o_sz_h * box1_o_sz_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&box2_out_buf, img_n * 64  * box2_o_sz_h * box2_o_sz_w * sizeof(float)) );
    cudaErrchk( cudaMalloc((void **)&box3_out_buf, img_n * 128 * box3_o_sz_h * box3_o_sz_w * sizeof(float)) );

    uint32 w_acc_sz = 0;
    for(auto i = 0; i <= 18; i++){
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

    cudnnConvolutionFwdAlgo_t conv2d_algo;
    cudnnErrchk(cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
                                                    tensor_in_desc,
                                                    kernel_desc,
                                                    conv2d_desc,
                                                    tensor_out_desc,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    0, /*memoryLimitInBytes=*/
                                                    &conv2d_algo));

    size_t workspace_bytes = 0;
    cudnnErrchk(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                        tensor_in_desc,
                                                        kernel_desc,
                                                        conv2d_desc,
                                                        tensor_out_desc,
                                                        conv2d_algo,
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
                                        conv2d_desc, conv2d_algo,
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
        alpha = 1, beta = 1;
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

    //box 1
    conv2d(conv_ch[1], conv_sz[1], n_conv[1], layer_buf1, layer_buf2,   weights_d[1], tmp_tensor_desc2, tmp_tensor_desc1, true);
    // printf_layer_io(tmp_tensor_desc2, tmp_tensor_desc1);

    conv2d(conv_ch[2], conv_sz[2], n_conv[2], layer_buf2, box1_out_buf, weights_d[2], tmp_tensor_desc1, tmp_tensor_desc2, true);
    // printf_layer_io(tmp_tensor_desc1, tmp_tensor_desc2);

    maxpooling(box1_out_buf, layer_buf1, tmp_tensor_desc2, tmp_tensor_desc1);
    // printf_layer_io(tmp_tensor_desc2, tmp_tensor_desc1);
    
    // box 2
    conv2d(conv_ch[3], conv_sz[3], n_conv[3], layer_buf1, layer_buf2,   weights_d[3], tmp_tensor_desc1, tmp_tensor_desc2, true);
    // printf_layer_io(tmp_tensor_desc1, tmp_tensor_desc2);

    conv2d(conv_ch[4], conv_sz[4], n_conv[4], layer_buf2, box2_out_buf, weights_d[4], tmp_tensor_desc2, tmp_tensor_desc1, true);
    // printf_layer_io(tmp_tensor_desc2, tmp_tensor_desc1);

    maxpooling(box2_out_buf, layer_buf1, tmp_tensor_desc1, tmp_tensor_desc2);
    // printf_layer_io(tmp_tensor_desc1, tmp_tensor_desc2);

    // box 3
    conv2d(conv_ch[5], conv_sz[5], n_conv[5], layer_buf1, layer_buf2,   weights_d[5], tmp_tensor_desc2, tmp_tensor_desc1, true);
    // printf_layer_io(tmp_tensor_desc2, tmp_tensor_desc1);

    conv2d(conv_ch[6], conv_sz[6], n_conv[6], layer_buf2, box3_out_buf, weights_d[6], tmp_tensor_desc1, tmp_tensor_desc2, true);
    // printf_layer_io(tmp_tensor_desc1, tmp_tensor_desc2);

    maxpooling(box3_out_buf, layer_buf1, tmp_tensor_desc2, tmp_tensor_desc1);
    // printf_layer_io(tmp_tensor_desc2, tmp_tensor_desc1);

    // intermediate 
    conv2d(conv_ch[7], conv_sz[7], n_conv[7], layer_buf1, layer_buf2, weights_d[7], tmp_tensor_desc1, tmp_tensor_desc2, true);
    // printf_layer_io(tmp_tensor_desc1, tmp_tensor_desc2);

    cudaErrchk( cudaMemcpy(img_out, layer_buf2, 128 * 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost) );

    auto predict_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to predict (inference)!\n", \
           chrono::duration_cast<chrono::microseconds>(predict_ed - predict_st ).count()/1000.);
}